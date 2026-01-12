import warnings
from math import gcd
from typing import Optional

import torch


def resolve_group_size(
    weight: torch.Tensor | None, requested_group_size: Optional[int]
) -> Optional[int]:
    """Resolve a safe quantization group size for the provided weight tensor.

    Prefers the requested group size when valid, otherwise selects the closest
    compatible divisor (or the full tensor size).
    """

    if weight is None:
        # Defer until weights are available; respect the requested value.
        return requested_group_size

    last_dim = weight.shape[-1]
    if requested_group_size in (-1, None) or requested_group_size == last_dim:
        return last_dim

    if requested_group_size is not None and requested_group_size <= 0:
        warnings.warn(
            "Invalid non-positive group size provided; defaulting to per-tensor quantization.",
            UserWarning,
        )
        return last_dim

    if requested_group_size is not None and last_dim % requested_group_size == 0:
        return requested_group_size

    if requested_group_size is None:
        return last_dim

    fallback = gcd(last_dim, requested_group_size)
    if fallback > 1:
        warnings.warn(
            (
                "Adjusted quantization group size from "
                f"{requested_group_size} to {fallback} to match layer dimension {last_dim}."
            ),
            UserWarning,
        )
        return fallback

    warnings.warn(
        (
            f"Requested group size {requested_group_size} is incompatible with "
            f"layer dimension {last_dim}; defaulting to per-tensor quantization."
        ),
        UserWarning,
    )
    return last_dim


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for level in levels[:-1]:
            if level.isdigit():
                mod_ = mod_[int(level)]
            else:
                mod_ = getattr(mod_, level)
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_named_linears(module, type_filter):
    return {name: m for name, m in module.named_modules() if isinstance(m, type_filter)}
