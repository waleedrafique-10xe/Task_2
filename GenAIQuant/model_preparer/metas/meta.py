from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from ..utils import ModelType


@dataclass
class ArchitectureMeta:
    architecture_class: Type
    model_type: ModelType
    layers: str
    embedding: str  # TOOD: Should be a list
    lm_head: str
    rope: str  # TOOD: Should be a list
    model_norm: str  # TOOD: Should be a list
    base_modules: list[str]
    layers_modules: list[list[str]]
    offline_rotations: dict[str, list[str]]
    online_rotations: dict[str, list[str]]
    layernorm_fuses: dict[str, dict[str, list[str]] | list[str]]
    vision_layers: str | None = None
    vision_layers_modules: list[list[str]] | None = None
    language_model_name: str | None = None
    vision_model_name: str | None = None
