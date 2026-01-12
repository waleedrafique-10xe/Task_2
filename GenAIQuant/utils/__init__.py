from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, Type

import torch
from platformdirs import user_cache_dir

CACHE_DIR = user_cache_dir("GenAIQuant")


if TYPE_CHECKING:
    from pydantic import BaseModel


def print_config(config: BaseModel, name: str | None = None):
    """Print a formatted representation of a configuration object.

    Args:
        config: A Pydantic BaseModel configuration instance to print
    """

    def _print_dict(data: dict, indent: int = 0):
        """Recursively print nested dictionary with proper indentation."""
        for key, value in data.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                _print_dict(value, indent + 1)
            elif isinstance(value, list):
                print("  " * indent + f"{key}:")
                for item in value:
                    if isinstance(item, dict):
                        _print_dict(item, indent + 1)
                    else:
                        print("  " * (indent + 1) + f"- {item}")
            elif isinstance(value, Type):
                print("  " * indent + f"{key}: {value.__name__}")
            else:
                print("  " * indent + f"{key}: {value}")

    print(f"\n{'=' * 60}")
    print(
        f"{config.__class__.__name__} Configuration"
        if name is None
        else name.capitalize()
    )
    print(f"{'=' * 60}")
    _print_dict(config.model_dump())
    print(f"{'=' * 60}\n")


def get_nested_attr(obj: object, attr: str) -> Any:
    return reduce(getattr, attr.split("."), obj)


def normalize_model_id(model_id: str):
    return model_id.replace("/", "--")


def iterative_map_to_device(
    inputs: list[torch.Tensor] | dict[str, torch.Tensor],
    device: str = "cpu",
) -> list[torch.Tensor] | dict[str, torch.Tensor]:
    """
    Move all tensors in the input collection to the specified device.

    Args:
        inputs (Union[List[torch.Tensor], Dict[str, torch.Tensor]]):
            Collection of PyTorch tensors to move to device
        device (str, optional): Target device to move tensors to.
            Defaults to "cpu".
            Can be "cpu", "cuda", "cuda:0", etc.

    Returns:
        list[torch.Tensor] | dict[str, torch.Tensor]:
            Collection of tensors moved to the specified device
            (same type as input)

    Example:
        >>> # With list
        >>> tensors_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        >>> gpu_tensors = all_to_device_generic(tensors_list, "cuda")

        >>> # With dictionary
        >>> tensors_dict = {"input": torch.tensor([1, 2, 3]),
                            "target": torch.tensor([4, 5, 6])}
        >>> gpu_tensors = all_to_device_generic(tensors_dict, "cuda")
    """

    if isinstance(inputs, list):
        return [tensor.to(device) for tensor in inputs]
    elif isinstance(inputs, dict):
        return {key: tensor.to(device) for key, tensor in inputs.items()}
    elif isinstance(inputs, tuple):
        return tuple(tensor.to(device) for tensor in inputs)
    else:
        raise TypeError(
            f"Unsupported input type: {type(inputs)}. Expected list or dict."
        )


def recursive_map_to_device(
    inputs: list[Any] | dict[str, Any] | tuple | torch.Tensor,
    device: str = "cpu",
) -> list[torch.Tensor] | dict[str, torch.Tensor] | tuple | torch.Tensor:
    """
    Recursively move all tensors in nested collections to the specified device in-place.

    Args:
        inputs: Collection of PyTorch tensors (can be nested lists/dicts/tuples)
        device (str, optional): Target device to move tensors to. Defaults to "cpu".

    Returns:
        Collection of tensors moved to the specified device (same structure as input)
        Note: Lists and dicts are modified in-place, tuples return new instances.
    """

    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)

    elif isinstance(inputs, dict):
        for key, value in inputs.items():
            inputs[key] = recursive_map_to_device(value, device)
        return inputs

    elif isinstance(inputs, list):
        for i, item in enumerate(inputs):
            inputs[i] = recursive_map_to_device(item, device)
        return inputs

    elif isinstance(inputs, tuple):
        # Tuples are immutable, so we always return a new tuple
        return tuple(recursive_map_to_device(item, device) for item in inputs)

    else:
        # For non-tensor, non-container objects, return as-is
        return inputs
