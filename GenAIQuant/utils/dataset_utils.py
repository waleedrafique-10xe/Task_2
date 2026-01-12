import itertools
import os
import random
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import ProcessorMixin

from ..logger import logger


class DatasetProvider:
    """
    A unified class to load datasets from either the Hugging Face hub or local files.

    Supports:
    - Named Hugging Face datasets with internal configuration resolution.
    - Local datasets in CSV or JSON format.
    - Optional sampling and shuffling.
    - Access to dataset features.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        """Initialize the provider with empty dataset state."""
        self._dataset = None
        self._dataset_name = None
        cache_root = cache_dir or os.environ.get(
            "GenAIQuant_DATASET_CACHE", "./cache/hf_datasets"
        )
        self.cache_dir = Path(cache_root).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        name_or_path: str,
        processor: ProcessorMixin | None = None,
        num_samples: Optional[int] = None,
        split: Optional[str] = None,
        shuffle: bool = False,
        oversample_factor: int = 3,
    ) -> Dataset | DatasetDict:
        """
        Load a dataset from Hugging Face or a local path and provide to user as required

        Args:
            name_or_path (str): Name of the dataset (e.g., 'wikitext') or local file path.
            num_samples (int, optional): If set, limits the dataset to the first `num_samples` rows.
            split (str, optional): Specific split to load (e.g., "train", "test", "train[:10]", "test[:3%]").
            shuffle (bool, optional): If True and num_samples is set, shuffles before sampling.

        Returns:
            Dataset or DatasetDict: Loaded dataset (single split or multiple splits).
        """

        self._dataset_name = name_or_path
        data_type: str = "text"
        # Determine source: Hugging Face or local
        if self._is_huggingface_dataset(name_or_path):
            ds, data_type = self._load_huggingface_dataset(name_or_path, split)
        elif Path(name_or_path).exists():
            ds = self._load_local_dataset(name_or_path, split)
        else:
            raise ValueError(
                f"'{name_or_path}' is neither a supported dataset name nor a valid path."
            )
        if data_type == "visual":
            # grab a bit extra
            self._dataset = ds
            if num_samples != None:
                sampled_iter = itertools.islice(ds, num_samples * oversample_factor)
            else:
                raise ValueError(
                    f"Expected value for num_samples is integer for sampled_iter but found {num_samples}"
                )
            sampled_list = list(sampled_iter)
            random.shuffle(sampled_list)

            logger.info(f"Downloaded {len(sampled_list)} samples")

            calib_data = []

            for sample in sampled_list:
                if len(calib_data) == num_samples:
                    break
                images = []
                messages = []

                for turn in sample["messages"]:  # type: ignore
                    content = []
                    for item in turn["content"]:
                        if item["type"] == "text" and item["text"] is not None:
                            content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image":
                            # collect the PIL image and insert an image placeholder
                            images.append(sample["images"][item["index"]])
                            content.append({"type": "image"})
                    messages.append({"role": turn["role"], "content": content})

                if processor.chat_template is None:
                    text_parts = []
                    for message in messages:
                        for item in message["content"]:
                            if item["type"] == "text":
                                text_parts.append(item["text"])
                            elif item["type"] == "image":
                                text_parts.append(
                                    processor.tokenizer.special_tokens_map["boi_token"]
                                )
                    prompt = "\n\n".join(text_parts)

                else:
                    prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                try:
                    inputs = processor(
                        text=prompt, images=images, padding=True, return_tensors="pt"
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

                data["token_type_ids"] = torch.nn.functional.pad(
                    data["token_type_ids"],
                    (0, max_seqlen - data["token_type_ids"].shape[1]),
                    value=0,
                )
                # ^ we assume token_type_ids is a mask that determines which tokens
                # come from the image. Everything else (including padded tokens)
                # should be zero as well

            assert len(calib_data) == num_samples
            logger.info(f"Obtained {num_samples} samples from {self._dataset_name}")
            return calib_data
        else:
            # Optionally shuffle and select a sample
            if num_samples is not None:
                if not isinstance(ds, Dataset):
                    raise TypeError(
                        "Sampling only supported for single split Dataset, not DatasetDict."
                    )
                if shuffle:
                    print(shuffle)
                    ds = ds.shuffle()
                    print(num_samples)
                ds = ds.select(range(min(num_samples, len(ds))))

            self._dataset = ds
            self._dataset_name = name_or_path
            return ds

    def _is_huggingface_dataset(self, name: str) -> bool:
        """
        Check if the dataset name is in the supported Hugging Face config map.

        Args:
            name (str): Dataset name to check.

        Returns:
            bool: True if it's a known Hugging Face dataset, False otherwise.
        """
        return name in self._get_dataset_config_map()

    def _get_dataset_config_map(self) -> dict:
        """
        Internal mapping of friendly dataset names to actual Hugging Face names/configs.

        Returns:
            dict: Mapping of user-friendly names to (dataset_name, config_name).
        """
        return {
            "wikitext": ["wikitext", "wikitext-2-raw-v1", "text"],
            "c4": ["allenai/c4", "en/c4-validation.00000-of-00008.json.gz", "text"],
            "pg19": ["pg19", None, "text"],
            # NOTE: `togethercomputer/RedPajama-Data-1T` relies on a dataset script
            # which is no longer supported starting with `datasets>=3`. We instead
            # point to a hosted subset that exposes raw JSON files so the latest
            # datasets releases can load it without remote scripts.
            "redpajama": [
                "konwoo/RedPajama-Data-1T-Sample-subset10000",
                None,
                "text",
            ],
            "llava-instruct-mix-vsft": [
                "HuggingFaceH4/llava-instruct-mix-vsft",
                None,
                "visual",
            ],
            # Add more mappings here
        }

    def _resolve_dataset_config(self, name: str) -> list:
        """
        Resolve a friendly name to its (dataset_name, config_name) tuple.

        Args:
            name (str): Friendly dataset name.

        Returns:
            list: Hugging Face dataset, config name and dataset type.

        Raises:
            ValueError: If the name is not in the config map.
        """

        config_map = self._get_dataset_config_map()
        if name not in config_map:
            raise ValueError(f"Dataset '{name}' is not registered in config map.")
        return config_map[name]

    def _load_huggingface_dataset(
        self, name: str, split: Optional[str]
    ) -> Dataset | DatasetDict:
        """
        Load a dataset from Hugging Face using a resolved name and config.

        Args:
            name (str): Friendly name registered in config map.
            split (str, optional): Desired split to load.

        Returns:
            Dataset or DatasetDict: Loaded Hugging Face dataset.

        Raises:    random.shuffle(sampled_list)

            ValueError: If the split is invalid for the dataset.
        """

        hf_name, hf_config, data_type = self._resolve_dataset_config(name)
        try:
            if hf_config is not None and hf_config.endswith(".json.gz"):
                return [
                    load_dataset(
                        hf_name,
                        data_files=hf_config,
                        split=split if split else None,
                        cache_dir=str(self.cache_dir),
                    ),
                    data_type,
                ]
            else:
                _stream = True if data_type == "visual" else False
                return [
                    load_dataset(
                        hf_name,
                        hf_config,
                        split=split if split else None,
                        cache_dir=str(self.cache_dir),
                        streaming=_stream,
                    ),
                    data_type,
                ]

        except ValueError as e:
            raise ValueError(f"Invalid split '{split}' for dataset '{name}'.") from e

    def _load_local_dataset(
        self, path: str, split: Optional[str]
    ) -> Dataset | DatasetDict:
        """
        Load a dataset from a local JSON or CSV file.

        Args:
            path (str): Local path to the dataset file.
            split (str, optional): Optional split name to simulate (e.g., "train").

        Returns:
            Dataset or DatasetDict: Loaded dataset from local file.

        Raises:
            ValueError: If file extension is not supported.
            RuntimeError: If dataset loading fails.
        """

        ext = Path(path).suffix.lower()
        if ext not in [".json", ".csv"]:
            raise ValueError(
                f"Unsupported local file type: {ext}. Only .json and .csv are supported."
            )
        file_format = "json" if ext == ".json" else "csv"
        try:
            if split:
                return load_dataset(
                    file_format,
                    data_files={split: path},
                    split=split,
                    cache_dir=str(self.cache_dir),
                )
            else:
                return load_dataset(
                    file_format, data_files=path, split="train", cache_dir=str(self.cache_dir)
                )
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load local dataset from: {path}") from e

    @property
    def features(self):
        """
        Get the features (column schema) of the currently loaded dataset.

        Returns:
            dict: Features dictionary for the current dataset split.

        Raises:
            RuntimeError: If no dataset has been loaded yet.
        """
        if self._dataset is None:
            raise RuntimeError("No dataset loaded. Call `.get()` first.")
        if isinstance(self._dataset, Dataset):
            return self._dataset.features
        elif isinstance(self._dataset, DatasetDict):
            first_split = next(iter(self._dataset))
            return self._dataset[first_split].features
        else:
            logger.warning("Unknown dataset format with unknow features.")
