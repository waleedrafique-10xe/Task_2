from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from GenAIQuant.logger import logger
from GenAIQuant.model_preparer.utils import ModelType
from GenAIQuant.utils import get_nested_attr
from GenAIQuant.utils.dtype import parse_dtype

if TYPE_CHECKING:
    from .metas import ArchitectureMeta


class Model:
    """
    Represents a prepared model for quantization with its components and exported graph.

    Attributes:
        model_id (str): Identifier for the model.
        model (PreTrainedModel | nn.Module): The original model.
        model_config (PretrainedConfig): Configuration of the model.
        layers (nn.ModuleList | object): Decoder layers of the model.
        embedding (nn.Module | object): Embedding layer of the model.
        layernorm_fused (bool): Whether layer normalization is fused.
        dummy_inputs: Sample inputs for the model.
        name_to_module_type (MappingProxyType): Mapping from module names to their
        types.
    """

    def __init__(
        self,
        model_id: str,
        model: PreTrainedModel | nn.Module,
        layers: nn.ModuleList,
        embedding: nn.Module,
        rope: nn.Module,
        model_norm,
        lm_head: nn.modules.linear.Linear,
        model_meta: ArchitectureMeta,
        model_config: PretrainedConfig,
        vision_layers: nn.ModuleList | None = None,
        layernorm_fused: bool = True,
        device: str = "cpu",
        # device_map: dict[str, int | str | device] | OrderedDict | None = None,
    ):
        """
        Initialize a Model instance.

        Args:
            model_id (str): Identifier for the model.
            model (PreTrainedModel | nn.Module): The original model.
            layers (nn.ModuleList): Decoder layers of the model.
            embedding (nn.Module): Embedding layer of the model.
            model_norm: Model Norm.
            model_config (PretrainedConfig): Configuration of the model.
            lm_head: nn.modules.linear.Linear: Language model head.
            layernorm_fused (bool, optional): Whether layer normalization is fused.
                                       Defaults to True.
        """

        self.model_id = model_id
        self.model = model
        self.model_config = model_config
        self.meta = model_meta
        self.layers = layers
        self.embedding = embedding
        self.rope = rope
        self.model_norm = model_norm
        self.lm_head = lm_head
        self.meta = model_meta
        self.layernorm_fused = layernorm_fused
        self.dummy_inputs = model.dummy_inputs
        # self.device_map = device_map

        self.vision_layers = vision_layers

        self.device = device

    def gpu(self):
        # if self.device_map is None:
        #     logger.warning("No device map found, keeping model in CPU")
        #     return
        logger.info(f"Mapping model to gpu device: {self.device}")
        self.model = self.model.to(device=self.device)
        return self

    def cpu(self):
        self.model = self.model.cpu()
        return self

    @property
    def hidden_size(self):
        match self.meta.model_type:
            case ModelType.LLM:
                return self.model_config.hidden_size
            case ModelType.VLM:
                return self.model_config.text_config.hidden_size

    @property
    def is_multimodal(self) -> bool:
        return (
            self.meta.model_type == ModelType.VLM
            and self.meta.vision_layers is not None
        )

    @staticmethod
    def _config_dtype(config) -> torch.dtype | None:
        if config is None:
            return None
        value = config.torch_dtype if hasattr(config, "torch_dtype") else None
        return parse_dtype(value)

    @property
    def language_dtype(self) -> torch.dtype | None:
        dtype = self._config_dtype(self.model_config)
        if dtype is not None:
            return dtype
        param = next(self.model.parameters(), None)
        return param.dtype if param is not None else None

    @property
    def vision_dtype(self) -> torch.dtype | None:
        if not self.is_multimodal:
            return None
        text_cfg = (
            self.model_config.text_config
            if hasattr(self.model_config, "text_config")
            else None
        )
        vision_cfg = (
            self.model_config.vision_config
            if hasattr(self.model_config, "vision_config")
            else None
        )
        dtype = self._config_dtype(vision_cfg)
        if dtype is not None:
            return dtype
        dtype = self._config_dtype(text_cfg)
        if dtype is not None:
            return dtype
        return self.language_dtype

    @property
    def vision_model_prefix(self) -> str | None:
        if not self.is_multimodal:
            return None

        base_name = self.meta.vision_model_name
        if base_name:
            return base_name if base_name.startswith("model.") else f"model.{base_name}"

        layers_name = self.meta.vision_layers
        if layers_name:
            cleaned = layers_name[:-2] if layers_name.endswith(".#") else layers_name
            segments = cleaned.split(".")
            if len(segments) > 1:
                segments = segments[:-1]
            if (
                segments
                and segments[-1] in {"encoder", "decoder"}
                and len(segments) > 1
            ):
                segments = segments[:-1]
            return ".".join(segments)
        return None

    def get_vision_module(self, root: PreTrainedModel | nn.Module | None = None):
        if not self.is_multimodal:
            return None
        candidates = _vision_lookup_candidates(self.meta)
        target_root = root if root is not None else self.model
        for candidate in candidates:
            module = _try_get_nested(target_root, candidate)
            if module is not None:
                return module
        logger.warning(
            "Vision module could not be resolved for model_id=%s using candidates %s.",
            self.model_id,
            candidates,
        )
        return None


def _expand_prefix_candidates(base: str) -> list[str]:
    prefixes = []
    if base.startswith("model."):
        prefixes.append(base)
        suffix = base[len("model.") :]
        if suffix:
            prefixes.append(suffix)
    else:
        prefixed = f"model.{base}"
        prefixes.append(prefixed)
        prefixes.append(base)
    return _unique_preserve_order(prefixes)


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _try_get_nested(obj: object, attr: str):
    try:
        return get_nested_attr(obj, attr)
    except AttributeError:
        return None


def _vision_lookup_candidates(meta: "ArchitectureMeta") -> list[str]:
    candidates: list[str] = []
    base_name = meta.vision_model_name
    if base_name:
        candidates.extend(_expand_prefix_candidates(base_name))

    layers_name = meta.vision_layers
    if layers_name:
        cleaned = layers_name[:-2] if layers_name.endswith(".#") else layers_name
        segments = cleaned.split(".")
        while segments:
            candidate = ".".join(segments)
            candidates.extend(_expand_prefix_candidates(candidate))
            segments = segments[:-1]
    return _unique_preserve_order(candidates)
