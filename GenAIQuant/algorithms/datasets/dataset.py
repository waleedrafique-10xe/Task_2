from typing import Iterable, Sequence
import tarfile
import itertools
import random

from datasets import load_dataset, Dataset, DatasetDict
from transformers import ProcessorMixin
import torch
from PIL import Image
from huggingface_hub import hf_hub_download


from GenAIQuant.logger import logger
from GenAIQuant.utils.dataset_utils import DatasetProvider


def _resolve_text_column(split: Dataset) -> str:
    """
    Pick the column that holds raw text. Prefer common names and fall back to the first string column.
    """
    preferred_columns = ("text", "content")
    for column in preferred_columns:
        if column in split.column_names:
            return column
    for name, feature in split.features.items():
        if getattr(feature, "dtype", None) in ("string", "large_string"):
            return name
    return split.column_names[0]

def _prepare_train_val_splits(
    dataset: Dataset | DatasetDict, seed: int = 0
) -> tuple[Dataset, Dataset]:
    """
    Return (train_split, val_split) no matter what the original dataset layout is.
    - If the dataset already has train + validation/test splits, use them.
    - If only a single split exists, create a deterministic split.
    """
    if isinstance(dataset, DatasetDict):
        train_split = dataset.get("train") or next(iter(dataset.values()))
        val_split = dataset.get("validation") or dataset.get("test")
        if val_split is None:
            split = train_split.train_test_split(test_size=0.1, seed=seed)
            train_split, val_split = split["train"], split["test"]
    elif isinstance(dataset, Dataset):
        if len(dataset) > 1:
            split = dataset.train_test_split(test_size=0.1, seed=seed)
            train_split, val_split = split["train"], split["test"]
        else:
            train_split = val_split = dataset
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")
    return train_split, val_split

def _concat_until_tokens(
    split: Dataset,
    text_column: str,
    tokenizer,
    target_tokens: int,
) -> str:
    """
    Accumulate examples until at least `target_tokens` tokens are reached.
    Avoids concatenating an entire massive dataset like RedPajama.
    """
    pieces = []
    token_budget = 0
    for row in split:
        text = str(row[text_column])
        pieces.append(text)
        token_budget += len(
            tokenizer.encode(text, add_special_tokens=False)
        )
        if token_budget >= target_tokens:
            break
    if not pieces:
        raise ValueError("Dataset split is empty; cannot build calibration corpus.")
    return "\n\n".join(pieces)

def _load_split(dataset_name: str, split: str) -> Dataset:
    try:
        return load_dataset(dataset_name, split=split)  # type: ignore[arg-type]
    except ValueError:
        logger.warning(
            "Split '%s' not found for %s. Falling back to 'val'.", split, dataset_name
        )
        return load_dataset(dataset_name, split="val")  # type: ignore[arg-type]

def _sample_examples(
    dataset: Dataset, target_size: int, rng: random.Random
) -> Sequence[dict]:
    if target_size >= len(dataset):
        indices = list(range(len(dataset)))
    else:
        indices = rng.sample(range(len(dataset)), target_size)
    return [dataset[int(idx)] for idx in indices]

def _process_examples(
    processor: ProcessorMixin,
    samples: Iterable[dict],
    expected: int,
    max_length: int | None,
) -> list[tuple[dict[str, torch.Tensor], None]]:
    return list(
        _iter_processed_examples(
            processor=processor,
            samples=samples,
            expected=expected,
            max_length=max_length,
        )
    )

def _process_single_example(
    processor: ProcessorMixin,
    sample: dict,
    max_length: int | None,
) -> dict[str, torch.Tensor] | None:
    try:
        prompt = _build_prompt(processor, sample.get("caption", ""))
        call_kwargs = {
            "text": prompt,
            "images": sample.get("image"),
            "padding": True,
            "return_tensors": "pt",
        }
        if max_length is not None:
            call_kwargs.update(
                {
                    "truncation": True,
                    "max_length": max_length,
                }
            )
        inputs = processor(**call_kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging for bad samples
        logger.warning("Skipping sample due to processor failure: %s", exc)
        return None
    cleaned = {
        key: value.detach().clone()
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    if not cleaned:
        return None
    return cleaned

def _iter_processed_examples(
    processor: ProcessorMixin,
    samples: Iterable[dict],
    expected: int,
    max_length: int | None,
) -> Iterable[tuple[dict[str, torch.Tensor], None]]:
    processed_count = 0
    for sample in samples:
        cleaned = _process_single_example(processor, sample, max_length)
        if cleaned is None:
            continue
        yield (cleaned, None)
        processed_count += 1
        if processed_count == expected:
            break
    if processed_count < expected:
        logger.warning(
            "Requested %d samples but only obtained %d after filtering.",
            expected,
            processed_count,
        )

def _build_prompt(processor: ProcessorMixin, caption: str) -> str:
    caption = caption or ""
    if hasattr(processor, "apply_chat_template") and getattr(
        processor, "chat_template", None
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": caption.strip()},
                ],
            }
        ]
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return caption.strip()


class DatasetUtils:
    """This class contains functions to produce datasets for all of the algorithms"""

                                                                        # PTQ functions

    @staticmethod
    def get_calib_dataset(datasetname, num_samples, split, seqlen, tokenizer):
        # TODO: We should take this out of the class and move to a place where
        # other algorithms might be able to take advantage

        valid_keys = {"text", "sentence"}
        dataset_key = None
        dataset_provider = DatasetProvider()
        dataset = dataset_provider.get(
            name_or_path=datasetname, num_samples=num_samples, split=split
        )

        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")

        train_loader = []
        random.seed(42)
        for key_ in valid_keys:
            if key_ in dataset.features:
                dataset_key = key_
                break

        if dataset_key is None:
            raise ValueError(
                f"Invalid key for dataset. A valid key should befrom {dataset.features}"
            )

        raw_text = tokenizer("\n\n".join(dataset[key_]), return_tensors="pt")

        if seqlen >= raw_text.input_ids.shape[1]:
            raise ValueError(
                f"Sequence length {seqlen} is too long for input"
                f"length {raw_text.input_ids.shape[1]}"
            )

        for _ in range(num_samples):
            i = random.randint(0, raw_text.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = raw_text.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            train_loader.append((inp, tar))
        return train_loader

    @staticmethod
    def get_vlm_calib_dataset(
        processor,
        dataset_name: str,
        num_samples: int = 128,
        split: str = "train",
        oversample_factor: int = 3,
    ):
        logger.info(f"Obtaining dataset {dataset_name} from Hugging face")
        streamed_dataset = load_dataset(dataset_name, split=split, streaming=True)

        # grab a bit extra
        sampled_iter = itertools.islice(streamed_dataset, num_samples * oversample_factor)

        sampled_list = list(sampled_iter)
        random.shuffle(sampled_list)

        logger.info(f"Downloaded {len(sampled_list)} samples")

        calib_data = []

        # TODO: 6D nesting doesn''t look good
        for i, sample in enumerate(sampled_list):
            if len(calib_data) == num_samples:
                break
            images = []
            messages = []

            for turn in sample["messages"]:
                content = []

                for count, item in enumerate(turn["content"]):
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





                                                                        # EQAT functions

    @staticmethod
    def get_dataset(
        tokenizer,
        dataset_name: str = "wikitext",
        train_size: int = 128,
        val_size: int = 32,
        seed: int = 0,
        seqlen: int = 2048,
        test_only: bool = False,
    ):
        dataset = DatasetProvider()
        dataset_ = dataset.get(name_or_path=dataset_name)
        train_data, val_data = _prepare_train_val_splits(dataset_, seed=seed)
        text_column = _resolve_text_column(train_data)
        target_val_tokens = max(1, val_size) * seqlen
        val_corpus = _concat_until_tokens(
            val_data, text_column, tokenizer=tokenizer, target_tokens=target_val_tokens
        )
        test_text = tokenizer(val_corpus, return_tensors="pt")
        if test_only:
            return test_text

        target_train_tokens = max(1, train_size + val_size) * seqlen
        train_corpus = _concat_until_tokens(
            train_data,
            text_column,
            tokenizer=tokenizer,
            target_tokens=target_train_tokens,
        )
        train_text = tokenizer(train_corpus, return_tensors="pt")

        if train_text.input_ids.shape[1] <= seqlen + 1:
            raise ValueError(
                f"Calibration corpus is too small for seqlen={seqlen}. "
                "Provide a larger dataset or reduce `training_seqlen`."
            )

        random.seed(seed)
        train_loader = []
        validation_loader = []

        val_sample_ratio = (
            0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
        )
        for _ in range(train_size):
            i = random.randint(
                0, int(train_text.input_ids.shape[1] * val_sample_ratio) - seqlen - 1
            )
            j = i + seqlen
            inp = train_text.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            train_loader.append((inp, tar))
        valloader = []
        for _ in range(val_size):
            i = random.randint(
                int(train_text.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
                train_text.input_ids.shape[1] - seqlen - 1,
            )
            j = i + seqlen
            inp = train_text.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            validation_loader.append((inp, tar))
        return train_loader, validation_loader

    @staticmethod
    def get_vl_dataset(
        processor: ProcessorMixin,
        dataset_name: str,
        train_size: int,
        val_size: int,
        seed: int = 0,
        train_split: str = "train",
        val_split: str = "validation",
        max_length: int | None = None,
    ) -> tuple[
        list[tuple[dict[str, torch.Tensor], None]],
        list[tuple[dict[str, torch.Tensor], None]],
    ]:
        """Prepare multimodal calibration data for vision-language models."""

        if dataset_name == "AIMClab-RUC/COCO-CN":
            loader = _CocoCnDataset(processor=processor, max_length=max_length)
            train_batch = loader.prepare_split(
                split=train_split,
                requested_size=train_size,
                seed=seed,
            )
            val_batch = loader.prepare_split(
                split=val_split,
                requested_size=val_size,
                seed=seed,
            )
            return train_batch, val_batch

        train_dataset = _load_split(dataset_name, train_split)
        val_dataset = _load_split(dataset_name, val_split)

        rng = random.Random(seed)
        train_samples = _sample_examples(train_dataset, train_size, rng)
        val_samples = _sample_examples(val_dataset, val_size, rng)

        train_batch = _process_examples(processor, train_samples, train_size, max_length)
        val_batch = _process_examples(processor, val_samples, val_size, max_length)

        return train_batch, val_batch


class _CocoCnDataset:
    """Helper to prepare AIMClab-RUC/COCO-CN samples backed by MS-COCO images."""

    _REPO_ID = "AIMClab-RUC/COCO-CN"
    _ARCHIVE = "coco-cn-version1805v1.1.tar.gz"
    _ROOT = "coco-cn-version1805v1.1"
    _CAPTION_FILES = (
        f"{_ROOT}/imageid.human-written-caption.txt",
        f"{_ROOT}/imageid.manually-translated-caption.txt",
    )
    _SPLIT_FILES = {
        "train": f"{_ROOT}/coco-cn_train.txt",
        "val": f"{_ROOT}/coco-cn_val.txt",
        "test": f"{_ROOT}/coco-cn_test.txt",
    }
    _IMAGE_BASE = "https://images.cocodataset.org"

    def __init__(self, processor: ProcessorMixin, max_length: int | None):
        self._processor = processor
        self._max_length = max_length
        self._split_ids: dict[str, list[str]] = {}
        self._captions: dict[str, list[str]] = {}
        self._load_metadata()

    def prepare_split(
        self,
        *,
        split: str,
        requested_size: int,
        seed: int,
    ) -> list[tuple[dict[str, torch.Tensor], None]]:
        """Load and process a COCO-CN split."""
        return list(
            self.iter_split(split=split, requested_size=requested_size, seed=seed)
        )

    def iter_split(
        self,
        *,
        split: str,
        requested_size: int,
        seed: int,
    ) -> Iterable[tuple[dict[str, torch.Tensor], None]]:
        normalized = self._normalize_split(split)
        available_ids = self._split_ids.get(normalized, [])
        if not available_ids:
            logger.warning(
                "COCO-CN split '%s' is empty. Requested %d samples.",
                split,
                requested_size,
            )
            return []

        rng = random.Random(seed)
        if requested_size >= len(available_ids):
            chosen_ids = available_ids.copy()
            rng.shuffle(chosen_ids)
        else:
            chosen_ids = rng.sample(available_ids, requested_size)

        raw_samples = self._build_samples(chosen_ids, expected=requested_size)

        return _iter_processed_examples(
            processor=self._processor,
            samples=raw_samples,
            expected=requested_size,
            max_length=self._max_length,
        )

    def _build_samples(
        self, image_ids: list[str], *, expected: int
    ) -> Iterable[dict]:
        for image_id in image_ids:
            base_id = self._base_image_id(image_id)
            caption = self._pick_caption(image_id)
            if caption is None:
                continue
            image = self._fetch_image(base_id)
            if image is None:
                continue
            yield {
                "image": image,
                "caption": caption,
            }
            if expected and expected > 0:
                expected -= 1
                if expected == 0:
                    break

    def _pick_caption(self, image_id: str) -> str | None:
        captions = self._captions.get(image_id)
        if not captions:
            return None
        return captions[0]

    def _load_metadata(self):
        archive_path = hf_hub_download(
            repo_id=self._REPO_ID,
            filename=self._ARCHIVE,
            repo_type="dataset",
        )
        with tarfile.open(archive_path, "r:gz") as tar:
            self._captions = self._load_captions(tar)
            self._split_ids = self._load_split_ids(tar)

    def _load_captions(self, tar: tarfile.TarFile) -> dict[str, list[str]]:
        captions: dict[str, list[str]] = {}
        for file_name in self._CAPTION_FILES:
            try:
                member = tar.getmember(file_name)
            except KeyError:
                continue
            data = tar.extractfile(member)
            if data is None:
                continue
            for raw in data:
                line = raw.decode("utf-8").strip()
                if not line or "\t" not in line:
                    continue
                key, text = line.split("\t", 1)
                base_id = key.split("#", 1)[0]
                text = text.strip()
                if not text:
                    continue
                captions.setdefault(base_id, []).append(text)
        return captions

    def _load_split_ids(self, tar: tarfile.TarFile) -> dict[str, list[str]]:
        splits: dict[str, list[str]] = {name: [] for name in self._SPLIT_FILES}
        for split_name, file_name in self._SPLIT_FILES.items():
            try:
                member = tar.getmember(file_name)
            except KeyError:
                continue
            data = tar.extractfile(member)
            if data is None:
                continue
            ids = [line.decode("utf-8").strip() for line in data]
            ids = [line for line in ids if line]
            splits[split_name] = ids
        return splits

    def _normalize_split(self, split: str) -> str:
        lowered = (split or "").lower()
        if lowered in {"validation", "val"}:
            return "val"
        if lowered in {"test", "testing"}:
            return "test"
        return "train"

    @staticmethod
    def _base_image_id(image_id: str) -> str:
        return image_id.split("#", 1)[0] if "#" in image_id else image_id

    def _fetch_image(self, base_id: str) -> Image.Image | None:
        filename = f"{base_id}.jpg"
        parts = base_id.split("_")
        if len(parts) < 3:
            return None
        split_folder = parts[1].lower()
        url = f"{self._IMAGE_BASE}/{split_folder}/{filename}"
        try:
            with urlopen(url) as response:
                payload = response.read()
        except URLError as exc:  # pragma: no cover - network failures
            logger.warning("COCO-CN: failed to fetch %s: %s", url, exc)
            return None
        try:
            with Image.open(BytesIO(payload)) as img:
                return img.convert("RGB")
        except Exception as exc:  # pragma: no cover - corrupted payload handling
            logger.warning("COCO-CN: failed to decode %s: %s", url, exc)
            return None