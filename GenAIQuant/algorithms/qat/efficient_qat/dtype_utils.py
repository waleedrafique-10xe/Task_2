from __future__ import annotations

import torch
import torch.nn as nn

from GenAIQuant.logger import logger
from GenAIQuant.utils.dtype import preferred_dtype_for_device

__all__ = ["_coerce_model_dtype", "_vision_dtype_overrides"]


def _first_module_dtype(module: nn.Module | None) -> torch.dtype | None:
    if module is None:
        return None
    for param in module.parameters():
        return param.dtype
    for buffer in module.buffers():
        if hasattr(buffer, "dtype"):
            return buffer.dtype
    return None


def _vision_dtype_overrides(vision_module: nn.Module | None) -> dict[str, torch.dtype]:
    overrides: dict[str, torch.dtype] = {}
    dtype = _first_module_dtype(vision_module)
    if dtype is None:
        return overrides
    overrides["pixel_values"] = dtype
    overrides["pixel_values_videos"] = dtype
    return overrides


def _coerce_model_dtype(
    model: nn.Module,
    device: torch.device | str,
    preferred_dtype: torch.dtype | None = None,
) -> tuple[nn.Module, torch.dtype]:
    device = torch.device(device)
    target_dtype = preferred_dtype

    if not isinstance(target_dtype, torch.dtype):
        param = next(model.parameters(), None)
        target_dtype = (
            param.dtype
            if (param is not None and isinstance(param.dtype, torch.dtype))
            else None
        )

    if target_dtype is None:
        target_dtype = preferred_dtype_for_device(device)

    if device.type == "cpu" and target_dtype != torch.float32:
        logger.info(
            "MBQ: forcing float32 on CPU execution (requested %s)", target_dtype
        )
        target_dtype = torch.float32

    model = model.to(device=device, dtype=target_dtype)
    return model, target_dtype
