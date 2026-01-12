from unittest.mock import MagicMock

import numpy as np
import pytest
from torch import nn
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel

from GenAIQuant.evaluation.metrics.lmeval import LMEval


class DummyModel(PreTrainedModel):
    def __init__(self):
        config = AutoConfig.from_pretrained("gpt2")
        super().__init__(config)
        self.transformer = nn.Identity()

    def forward(self, *args, **kwargs):
        return MagicMock()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def sample_result():
    return {
        "results": {
            "task1": {
                "accuracy": 0.9,
                "alias": "something",
                "metric_x": "N/A",
                "f1": np.float64(0.75),
                "exact_match": None,
            },
            "task2": {
                "accuracy": np.array(0.8),
                "task_perplexity": np.float32(10.2),
                "exact_match": 0.6,
            },
        }
    }


# Constructor tests
def test_valid_str_task(model, tokenizer, tmp_path):
    obj = LMEval(model, tokenizer, "task1", 1, 10, None, str(tmp_path))
    assert obj._task == ["task1"]


def test_valid_list_task(model, tokenizer, tmp_path):
    obj = LMEval(model, tokenizer, ["task1", "task2"], 1, 10, None, str(tmp_path))
    assert obj._task == ["task1", "task2"]


def test_invalid_model(tokenizer, tmp_path):
    with pytest.raises(TypeError):
        LMEval("not_model", tokenizer, "task", 1, 10, None, str(tmp_path))


def test_invalid_tokenizer(model, tmp_path):
    with pytest.raises(TypeError):
        LMEval(model, "wrong", "task", 1, 10, None, str(tmp_path))


def test_invalid_task_type(model, tokenizer, tmp_path):
    with pytest.raises(TypeError):
        LMEval(model, tokenizer, 123, 1, 10, None, str(tmp_path))


def test_invalid_output_dir(model, tokenizer):
    with pytest.raises(TypeError):
        LMEval(model, tokenizer, "task", 1, 10, None, 123)


# Report generation tests
def test_generate_report_saves_json(model, tokenizer, sample_result, tmp_path):
    obj = LMEval(
        model, tokenizer, "task1", 1, 10, None, str(tmp_path), save_output=True
    )
    obj.generate_report(sample_result)
    assert (tmp_path / "result.json").exists()


def test_generate_report_no_save(model, tokenizer, sample_result, tmp_path):
    obj = LMEval(
        model, tokenizer, "task1", 1, 10, None, str(tmp_path), save_output=False
    )
    obj.generate_report(sample_result)
    assert not (tmp_path / "result.json").exists()


def test_generate_report_prints_table(
    model, tokenizer, sample_result, tmp_path, capsys
):
    obj = LMEval(model, tokenizer, "task1", 1, 10, None, str(tmp_path))
    obj.generate_report(sample_result)
    captured = capsys.readouterr()
    assert "Results saved in" in captured.out
