from __future__ import annotations

import itertools
import random
from pprint import pformat
from typing import TYPE_CHECKING

import torch
from datasets import load_dataset
from transformers.models import AutoTokenizer
from transformers.models.auto import AutoProcessor

from GenAIQuant.logger import logger
from GenAIQuant.model_preparer.utils import ModelType
from GenAIQuant.utils.dataset_utils import DatasetProvider

from .base import BitAllocation
from .collect_metrics import collect_metrics

if TYPE_CHECKING:
    from torch import Tensor

    from GenAIQuant.config import Config
    from GenAIQuant.interfaces.base import ModuleInterface
    from GenAIQuant.model_preparer.model import Model


def get_vlm_calib_dataset(
    processor,
    dataset_name: str,
    num_samples: int = 128,
    split: str = "train",
    oversample_factor: int = 3,
) -> list[dict[str, Tensor]]:
    logger.info(f"Obtaining dataset {dataset_name} from Hugging face")
    streamed_dataset = load_dataset(dataset_name, split=split, streaming=True)

    # grab a bit extra
    sampled_iter = itertools.islice(streamed_dataset, num_samples * oversample_factor)

    sampled_list = list(sampled_iter)
    random.shuffle(sampled_list)

    logger.info(f"Downloaded {len(sampled_list)} samples")

    calib_data = []

    for sample in sampled_list:
        if len(calib_data) == num_samples:
            break
        images = []
        messages = []

        for turn in sample["messages"]:
            content = []
            for item in turn["content"]:
                if item["type"] == "text" and item["text"] is not None:
                    content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    # collect the PIL image and insert an image placeholder
                    image = sample["images"][item["index"]]
                    images.append(image)
                    content.append(
                        {
                            "type": "image",
                            "resized_width": 336,
                            "resized_height": 336,
                            "image": image,
                        }
                    )
            messages.append({"role": turn["role"], "content": content})

        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            # inputs = processor(
            #     text=prompt, images=images, padding=True, return_tensors="pt"
            # )
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=prompt,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        except ValueError as e:
            logger.warning(
                f"Found an invalid image with error: {e}. Skipping this sample"
            )
            continue

        calib_data.append(inputs)

    max_seqlen = max([x["input_ids"].shape[1] for x in calib_data])
    pad_token_id = processor.tokenizer.pad_token_type_id

    for data in calib_data:
        data["input_ids"] = torch.nn.functional.pad(
            data["input_ids"],
            (0, max_seqlen - data["input_ids"].shape[1]),
            value=pad_token_id,
        )
        data["attention_mask"] = torch.nn.functional.pad(
            data["attention_mask"],
            (0, max_seqlen - data["attention_mask"].shape[1]),
            value=1,
        )
    # ^ assumes everything is 1 in mask. will fail assumption is violated

    # data["token_type_ids"] = torch.nn.functional.pad(
    #     data["token_type_ids"],
    #     (0, max_seqlen - data["token_type_ids"].shape[1]),
    #     value=0,
    # )
    # ^ we assume token_type_ids is a mask that determines which tokens
    # come from the image. Everything else (including padded tokens)
    # should be zero as well

    assert len(calib_data) == num_samples
    logger.info(f"Obtained {num_samples} samples from {dataset_name}")
    return calib_data


def get_calib_dataset(
    tokenizer,
    dataset_name: str,
    num_samples: int,
    split: str,
    seq_len: int,
) -> list[tuple[torch.Tensor, ...]]:
    valid_keys = {"text", "sentence"}
    dataset_key = None
    dataset_provider = DatasetProvider()
    dataset = dataset_provider.get(
        name_or_path=dataset_name, num_samples=num_samples, split=split
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    train_loader = []
    for key_ in valid_keys:
        if key_ in dataset.features:
            dataset_key = key_
            break

    if dataset_key is None:
        raise ValueError(
            f"Invalid key for dataset. A valid key should be from {dataset.features}"
        )

    raw_text = tokenizer("\n\n".join(dataset[key_]), return_tensors="pt")

    if seq_len >= raw_text.input_ids.shape[1]:
        raise ValueError(
            f"Sequence length {seq_len} is too long for input length"
            f" {raw_text.input_ids.shape[1]}"
        )

    for _ in range(num_samples):
        i = random.randint(0, raw_text.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = raw_text.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        train_loader.append((inp, tar))

    return train_loader


class ShortGpt(BitAllocation):
    """
    Implements a BitAllocation algorithm based on ShortGPT's Block Influence
    metric to allocated bits.

    Bit Allocation adopts the following steps:
        1. Assigns all layers to `config.target_avg_bits`
        2. Creates and loads a dataset for calibration
        3. Add observers to all decoder layers to record block importance
           for each decoder layer
        4. Determines the layers to prune with `config.n_prune_layers` parameter
        5. Sets the bitwidth of layers_to_prune to lowest bit widht from
           `config.bitwidth_options`

    Parameters not part of decoder layers are also set to `config.target_avg_bits`
    """

    def __init__(self, config: Config):
        """
        Initialize the ShortGpt class with configuration settings.

        Args:
            config (Config): Configuration object containing initialization parameters.
                             Expected to have the following attributes:
                             - device: computational device
                             - output_dir: directory for output files
                             - target_avg_bitwidth: target average bitwidth for quantization
                             - n_prune_layers: number of layers to prune
                             - bitwidth_options: list of possible bitwidth options

        Attributes:
            device: Computational device for processing
            output_dir: Directory path for storing output files
            target_bits: Target average bitwidth for quantization
            initial_bits: Initial bitwidth (set to target bitwidth)
            n_prune_layers: Number of layers to prune
            bit_options: Sorted list of available bitwidth options
        """

        super().__init__(config)

        self.device = config.device
        self.output_dir = config.output_dir

        self.target_bits = self.config.target_avg_bitwidth
        self.initial_bits = self.target_bits
        self.n_prune_layers = self.config.n_prune_layers

        # make sure bit options are sorted
        self.bit_options = sorted(self.config.bitwidth_options)

    def forward(self, interface: ModuleInterface, **kwargs):
        """
        Perform bit allocation for model layers based on their importance.

        Args:
            interface (ModelInterface): Interface containing the model to be quantized

        Returns:
            ModelInterface: Updated interface with bit allocation information

        Key steps:
        - Uses calibration dataset to measure layer importance
        - Sorts layers by their metrics
        - Reduces bitwidth for the least important layers
        - Sets default bitwidth for other layers
        - Handles edge cases like insufficient layers to prune

        Logs:
        - Debug: Layer importance metrics
        - Info: Final bit allocation
        - Warning: If requested pruning exceeds total layers
        """

        model: Model = interface.model

        model_type = model.meta.model_type

        calib_dataloader = None
        if model_type == ModelType.VLM:
            logger.info(f"Loading processor for {model.model_id}")
            processor = AutoProcessor.from_pretrained(model.model_id, use_fast=True)

            logger.info(f"Getting calibration data from {self.config.dataset_name}")
            calib_dataloader = get_vlm_calib_dataset(
                processor,
                dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
                num_samples=self.config.num_samples,
            )

        elif model_type == ModelType.LLM:
            logger.info(f"Loading tokenizer for {model.model_id}")

            tokenizer = AutoTokenizer.from_pretrained(model.model_id)

            logger.info(f"Getting calibration data from {self.config.dataset_name}")

            calib_dataloader: list[tuple[torch.Tensor, ...]] = get_calib_dataset(
                tokenizer=tokenizer,
                dataset_name=self.config.dataset_name,
                num_samples=self.config.num_samples,
                split="train",
                seq_len=self.config.seq_len,
            )

        assert calib_dataloader is not None

        if self.device != "cpu":
            model.gpu()

        metrics = collect_metrics(model, calib_dataloader)

        sorted_metrics = sorted(metrics.items(), key=lambda item: item[1])
        logger.info(f"Block importance: {pformat(sorted_metrics)}")

        if self.n_prune_layers > len(model.layers):
            logger.warning(
                "Number of layers to reduce to low bits is more than number"
                "of layers. Falling back to uniform bit width distribution"
            )
            layers_to_prune = []
        else:
            layers_to_prune = [
                name for name, _ in sorted_metrics[: self.n_prune_layers]
            ]

        bit_allocation = {}
        for name, paramter in model.model.named_parameters():
            # only hande weight parameters
            if not name.endswith(".weight"):
                continue

            name = name.replace(".weight", "")

            bits = (
                self.bit_options[0]  # set to smallest option
                if any(prefix in name for prefix in layers_to_prune)
                else self.initial_bits
            )

            bit_allocation[name] = bits

        interface.bit_allocation = bit_allocation
        model.cpu()

        return interface
