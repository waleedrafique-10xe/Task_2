from __future__ import annotations

import random
import tarfile
from io import BytesIO
from typing import Iterable, Sequence
from urllib.error import URLError
from urllib.request import urlopen

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ProcessorMixin

from GenAIQuant.logger import logger





def iter_vl_dataset(
    processor: ProcessorMixin,
    dataset_name: str,
    train_size: int,
    val_size: int,
    seed: int = 0,
    train_split: str = "train",
    val_split: str = "validation",
    max_length: int | None = None,
) -> tuple[
    Iterable[tuple[dict[str, torch.Tensor], None]],
    Iterable[tuple[dict[str, torch.Tensor], None]],
]:

    if dataset_name == "AIMClab-RUC/COCO-CN":
        loader = _CocoCnDataset(processor=processor, max_length=max_length)

        def train_iter():
            yield from loader.iter_split(
                split=train_split,
                requested_size=train_size,
                seed=seed,
            )

        def val_iter():
            yield from loader.iter_split(
                split=val_split,
                requested_size=val_size,
                seed=seed,
            )

        return train_iter(), val_iter()

    train_dataset = _load_split(dataset_name, train_split)
    val_dataset = _load_split(dataset_name, val_split)

    def _make_iter(dataset: Dataset, requested_size: int):
        rng = random.Random(seed)
        if requested_size >= len(dataset):
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
        else:
            indices = rng.sample(range(len(dataset)), requested_size)

        def _raw_samples():
            for idx in indices:
                yield dataset[int(idx)]

        yield from _iter_processed_examples(
            processor=processor,
            samples=_raw_samples(),
            expected=requested_size,
            max_length=max_length,
        )

    return _make_iter(train_dataset, train_size), _make_iter(val_dataset, val_size)









