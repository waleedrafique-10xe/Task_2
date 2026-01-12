from __future__ import annotations

from typing import Any

import torch

__all__ = ["parse_dtype", "preferred_dtype_for_device"]


def parse_dtype(value: Any) -> torch.dtype | None:
    """Convert common dtype representations into a torch.dtype."""
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        attr = value.lower().strip()
        if attr.startswith("torch."):
            attr = attr[6:]
        aliases = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if attr in aliases:
            return aliases[attr]
        if hasattr(torch, attr):
            candidate = getattr(torch, attr)
            if isinstance(candidate, torch.dtype):
                return candidate
    return None


def _supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except RuntimeError:
        return False
    return major >= 8


def preferred_dtype_for_device(device: torch.device | str) -> torch.dtype:
    device = torch.device(device)
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16 if _supports_bf16() else torch.float16
