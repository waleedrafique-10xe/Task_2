from __future__ import annotations

from typing import TYPE_CHECKING

from .metas import gemma3_meta, llama_meta, qwen2_5_vl_meta

if TYPE_CHECKING:
    from .metas.meta import ArchitectureMeta

SUPPORTED_ARCHITECTURES: dict[str, ArchitectureMeta] = {
    "llama": llama_meta,
    "gemma3": gemma3_meta,
    "qwen2_5_vl": qwen2_5_vl_meta,
}
