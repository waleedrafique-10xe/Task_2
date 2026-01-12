import os
import random
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from GenAIQuant.algorithms.ptq.gptq.gptq import Gptq, GPTQConfig
from GenAIQuant.model_preparer.model_name_mapping import SUPPORTED_ARCHITECTURES


@pytest.fixture
def mock_config(monkeypatch):
    # Patch GPTQConfig dictionary
    from GenAIQuant.algorithms.ptq.gptq.gptq import (
        GPTQConfig,
    )  # Replace with your actual module

    monkeypatch.setitem(GPTQConfig, "seqlen", 16)
    monkeypatch.setitem(GPTQConfig, "percdamp", 0.01)
    monkeypatch.setitem(GPTQConfig, "act_order", False)
    monkeypatch.setitem(GPTQConfig, "static_groups", False)
    monkeypatch.setitem(GPTQConfig, "split", {"mock-dataset": "train"})

    ptq_config = SimpleNamespace(
        datasetname="mock-dataset", numsamples=2, wbits=4, sym=True, groupsize=32
    )
    quantization_config = SimpleNamespace(ptq=ptq_config)

    return SimpleNamespace(
        device="cpu", output_dir="/tmp/mock_output", quantization=quantization_config
    )


@pytest.fixture
def instance(mock_config):
    return Gptq(config=mock_config)


# Tests for find_layers method


def test_find_layers_top_level(instance):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    model = Model()
    result = instance.find_layers(model)
    assert "linear" in result
    assert isinstance(result["linear"], nn.Linear)


def test_find_layers_nested(instance):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Conv2d(1, 1, 3), nn.Linear(4, 4))

    model = Model()
    result = instance.find_layers(model)
    assert "block.0" in result
    assert "block.1" in result
    assert isinstance(result["block.0"], nn.Conv2d)
    assert isinstance(result["block.1"], nn.Linear)


def test_find_layers_deeply_nested(instance):
    class DeepModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(4, 4), nn.Sequential(nn.Conv2d(1, 1, 3))
            )

    model = DeepModel()
    result = instance.find_layers(model)
    assert "layer1.0" in result
    assert "layer1.1.0" in result
    assert isinstance(result["layer1.0"], nn.Linear)
    assert isinstance(result["layer1.1.0"], nn.Conv2d)


def test_find_layers_multiple_same_type(instance):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4)
            self.linear2 = nn.Linear(4, 4)

    model = Model()
    result = instance.find_layers(model)
    assert "linear1" in result
    assert "linear2" in result
    assert len(result) == 2


def test_find_layers_no_matching_layers(instance):
    class NoMatchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4)
            self.relu = nn.ReLU()

    model = NoMatchModel()
    result = instance.find_layers(model)
    assert result == {}


def test_find_layers_empty_module(instance):
    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()

    model = EmptyModel()
    result = instance.find_layers(model)
    assert result == {}


def test_find_layers_custom_layer_type(instance):
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4)

    model = CustomModel()
    result = instance.find_layers(model, layers=[nn.LayerNorm])
    assert "norm" in result
    assert isinstance(result["norm"], nn.LayerNorm)


def test_find_layers_fully_qualified_names(instance):
    class DuplicateNames(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(nn.Linear(4, 4))
            self.block2 = nn.Sequential(nn.Linear(4, 4))

    model = DuplicateNames()
    result = instance.find_layers(model)
    assert "block1.0" in result
    assert "block2.0" in result
    assert isinstance(result["block1.0"], nn.Linear)
    assert isinstance(result["block2.0"], nn.Linear)


# calib_dataset tests


def make_mock_dataset(features_keys, texts):
    dataset = MagicMock()
    dataset.features = features_keys
    dataset.__getitem__.side_effect = lambda key: texts
    dataset.__len__.return_value = len(texts)
    return dataset


def test_calib_dataset_returns_expected_length(instance):
    texts = ["This is a sentence.", "Another sentence.", "Yet one more."]
    mock_dataset = make_mock_dataset({"text"}, texts)

    # Patch DatasetProvider.get to return our mock dataset
    with patch(
        "GenAIQuant.utils.dataset_utils.DatasetProvider.get", return_value=mock_dataset
    ):
        tokenizer = MagicMock()
        tokenizer.return_tensors = "pt"
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1] * 100])
        }  # dummy input_ids tensor

        # Patch tokenizer call to produce a mock with input_ids attribute
        tokenizer.return_value = SimpleNamespace(
            input_ids=torch.randint(0, 100, (1, 50))
        )

        # Patch tokenizer to return a SimpleNamespace with input_ids
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ):
            train_loader = instance.calib_dataset(
                datasetname="mock-dataset",
                num_samples=2,
                split="train",
                seqlen=10,
                tokenizer=tokenizer,
            )
    assert len(train_loader) == 2
    for inp, tar in train_loader:
        assert inp.shape[1] == 10
        # Target mask: all except last token are -100
        assert (tar[:, :-1] == -100).all()
        # Last token in target should not be masked
        assert (tar[:, -1] != -100).all()


def test_calib_dataset_raises_for_invalid_key(instance):
    # Dataset features missing valid keys
    mock_dataset = make_mock_dataset({"invalid_key"}, ["text1", "text2"])
    with patch(
        "GenAIQuant.utils.dataset_utils.DatasetProvider.get", return_value=mock_dataset
    ):
        tokenizer = MagicMock()
        with pytest.raises(ValueError, match="Invalid key for dataset"):
            instance.calib_dataset(
                datasetname="mock-dataset",
                num_samples=1,
                split="train",
                seqlen=5,
                tokenizer=tokenizer,
            )


def test_calib_dataset_uses_first_valid_key(instance):
    texts = ["sent1", "sent2", "sent3"]
    # Provide multiple valid keys; should use the first found
    mock_dataset = make_mock_dataset({"sentence", "text"}, texts)
    with patch(
        "GenAIQuant.utils.dataset_utils.DatasetProvider.get", return_value=mock_dataset
    ):
        tokenizer = MagicMock()
        tokenizer.return_value = SimpleNamespace(
            input_ids=torch.randint(0, 100, (1, 50))
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ):
            result = instance.calib_dataset(
                datasetname="mock-dataset",
                num_samples=1,
                split="train",
                seqlen=10,
                tokenizer=tokenizer,
            )
    assert isinstance(result, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in result)


def test_calib_dataset_handles_small_input_ids(instance):
    # Input_ids length is smaller than seqlen + 1 (should raise due to randint range)
    mock_dataset = make_mock_dataset({"text"}, ["short"])
    with patch(
        "GenAIQuant.utils.dataset_utils.DatasetProvider.get", return_value=mock_dataset
    ):
        tokenizer = MagicMock()
        # input_ids too short for seqlen slicing
        tokenizer.return_value = SimpleNamespace(
            input_ids=torch.randint(0, 100, (1, 5))
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ):
            with pytest.raises(ValueError):
                instance.calib_dataset(
                    datasetname="mock-dataset",
                    num_samples=1,
                    split="train",
                    seqlen=10,  # longer than input_ids length
                    tokenizer=tokenizer,
                )


def test_calib_dataset_random_seed_consistency(instance):
    texts = ["sentence one.", "sentence two.", "sentence three."]
    mock_dataset = make_mock_dataset({"text"}, texts)
    with patch(
        "GenAIQuant.utils.dataset_utils.DatasetProvider.get", return_value=mock_dataset
    ):
        tokenizer = MagicMock()
        tokenizer.return_value = SimpleNamespace(
            input_ids=torch.randint(0, 100, (1, 50))
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ):
            random.seed(42)
            result1 = instance.calib_dataset(
                datasetname="mock-dataset",
                num_samples=3,
                split="train",
                seqlen=10,
                tokenizer=tokenizer,
            )
            random.seed(42)
            result2 = instance.calib_dataset(
                datasetname="mock-dataset",
                num_samples=3,
                split="train",
                seqlen=10,
                tokenizer=tokenizer,
            )
    # Should be identical if random seed is reset
    for (inp1, tar1), (inp2, tar2) in zip(result1, result2):
        assert torch.equal(inp1, inp2)
        assert torch.equal(tar1, tar2)


def test_calib_dataset_seqlen_too_large_raises(instance):
    texts = ["Short sentence."]
    mock_dataset = make_mock_dataset({"text"}, texts)

    with patch(
        "GenAIQuant.utils.dataset_utils.DatasetProvider.get", return_value=mock_dataset
    ):
        tokenizer = MagicMock()
        # Pretend the tokenizer returns a tensor with only 5 tokens
        tokenizer.return_value = SimpleNamespace(
            input_ids=torch.randint(0, 100, (1, 5))
        )

        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ):
            with pytest.raises(ValueError, match="Sequence length 10 is too long"):
                instance.calib_dataset(
                    datasetname="mock-dataset",
                    num_samples=1,
                    split="train",
                    seqlen=10,
                    tokenizer=tokenizer,
                )


# test forward function

GPTQConfig["seqlen"] = 16
GPTQConfig["percdamp"] = 0.01
GPTQConfig["act_order"] = False
GPTQConfig["static_groups"] = False
GPTQConfig["split"] = {"dummy-dataset": "train"}


class DummyPTQConfig:
    def __init__(self):
        self.device = "cpu"
        self.output_dir = "./tmp"
        self.quantization = SimpleNamespace(
            ptq=SimpleNamespace(
                numsamples=2,
                wbits=4,
                sym=True,
                groupsize=128,
                datasetname="dummy-dataset",
            )
        )


@pytest.fixture(scope="module")
def real_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("yujiepan/llama-3-tiny-random")
    tokenizer = AutoTokenizer.from_pretrained("yujiepan/llama-3-tiny-random")
    return model, tokenizer


@pytest.fixture
def patched_gptq(real_model_and_tokenizer):
    model, tokenizer = real_model_and_tokenizer
    gptq = Gptq(DummyPTQConfig())

    # Patch `calib_dataset` to avoid loading real datasets
    def fake_calib_dataset(datasetname, num_samples, split, seqlen, tokenizer):
        text = ["Hello world! This is a test input with enough length."] * num_samples
        tokens = tokenizer("\n\n".join(text), return_tensors="pt")

        if tokens.input_ids.shape[1] <= seqlen:
            raise ValueError(
                f"Tokenized input length {tokens.input_ids.shape[1]} is not enough"
                f" for seqlen {seqlen}"
            )

        train_loader = []
        for _ in range(num_samples):
            i = random.randint(0, tokens.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = tokens.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            train_loader.append((inp, tar))
        return train_loader

    gptq.calib_dataset = fake_calib_dataset

    return gptq, model, tokenizer


def test_forward_with_real_model(patched_gptq):
    gptq, model, tokenizer = patched_gptq
    model_meta = SUPPORTED_ARCHITECTURES["llama"]
    model = SimpleNamespace(
        model=model,
        model_id="yujiepan/llama-3-tiny-random",
        model_config=model.config,
        layers=model.model.layers,
        meta=model_meta,
        embedding=model.model.embed_tokens,
    )
    interface = SimpleNamespace(model=model, quant_params={}, bit_allocation=None)

    args = SimpleNamespace(model="yujiepan/llama-3-tiny-random")
    result = gptq.forward(interface=interface, args=args)

    assert hasattr(result, "model")
    assert hasattr(result, "quant_params")
    assert isinstance(result.model.model, PreTrainedModel)

    # remove temporary output directory after test
    folder = "./tmp"
    if os.path.exists(folder):
        shutil.rmtree(folder)
