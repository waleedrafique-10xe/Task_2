from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Type

import torch
from torch import nn
from transformers.activations import ACT2CLS

from GenAIQuant.logger import logger

from .modules import PassThrough

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from GenAIQuant.config import QConfig
    from GenAIQuant.model_preparer.metas import ArchitectureMeta


SUPPORTED_ACTIVATION_CLASSES: tuple[Type] = tuple(
    x[0] if isinstance(x, tuple) else x for x in ACT2CLS.values()
)

QDQ_CLASSES = (
    nn.Linear,
    nn.Embedding,
    nn.Conv3d,
    PassThrough,
    *SUPPORTED_ACTIVATION_CLASSES,
)

INT32_SCALE = 2**14
N_BITS = 8
VISION_N_BITS = 8
GROUP_SIZE = 64


def get_dtype_info(n_bits=8, is_signed=True):
    dtype = torch.int8

    signed_dtypes = {8: torch.int8, 16: torch.int16, 32: torch.int32}
    unsigned_dtypes = {8: torch.uint8, 16: torch.int16, 32: torch.int32}

    dtype = (
        signed_dtypes.get(n_bits, torch.int8)
        if is_signed
        else unsigned_dtypes.get(n_bits, torch.uint8)
    )
    return dtype, torch.iinfo(dtype)


def qdq(tensor: torch.Tensor, scale: int, n_bits: int = 8, is_signed: bool = True):
    dtype, dtype_info = get_dtype_info(n_bits, is_signed)

    # Map tensor to float32 to avoid values becoming inf
    tensor_dtype = tensor.dtype
    tensor = tensor.float()

    quantized_tensor = torch.zeros_like(tensor, dtype=dtype)
    quantized_tensor = torch.clamp(
        (tensor * scale).round(), dtype_info.min, dtype_info.max
    )
    qdq_tensor = quantized_tensor / scale

    return qdq_tensor.to(tensor_dtype)


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def qdq_group(
    tensor: torch.Tensor,
    n_bits: int = 8,
    group_size: int | None = None,
    axis: int = -1,
    offset_enabled: bool = False,
    is_two_power: bool = True,
    signed: bool = True,
) -> torch.Tensor:
    tensor_shape = tensor.shape
    og_shape = tensor.shape

    group_dim_size = None
    pad_size = 0

    if group_size is not None:
        # We should do padding here

        group_dim_size = tensor_shape[axis]
        pad_size = group_size - (group_dim_size % group_size)
        if pad_size == group_size:
            pad_size = 0

        if axis == -1:
            pad = (0, pad_size)

            tensor = nn.functional.pad(tensor, pad, "constant", 0)
            tensor_shape = tensor.shape

            tensor = tensor.reshape(-1, group_size)
        else:
            # axis += len(shape)
            pad = (0, 0, 0, pad_size)

            tensor = nn.functional.pad(tensor, pad, "constant", 0)
            tensor_shape = tensor.shape

            shape = tensor.shape[axis + 1 :]
            tensor = tensor.reshape(-1, group_size, *shape)

    reduce_shape = [axis]
    xmin = tensor.amin(reduce_shape, keepdim=True)
    xmax = tensor.amax(reduce_shape, keepdim=True)

    EPSILON = 1.22443e-15
    signed_min, signed_max = -(2 ** (n_bits - 1)), (2 ** (n_bits - 1)) - 1
    unsigned_min, unsigned_max = 0, (2**n_bits) - 1

    if is_two_power:
        if offset_enabled:
            diff = xmax - xmin
            new_max = diff + EPSILON
            intPart = torch.floor(torch.log2(new_max)) + torch.ones_like(new_max)
            fracPart = (n_bits) * torch.ones_like(intPart) - intPart
            scale = (2**fracPart).pow(-1).clamp(min=1e-6, max=1e6)

            offset = round_ste(-xmin / scale)

            tensor_int = (round_ste(tensor / scale) + offset).clamp(
                unsigned_min, unsigned_max
            )
            tensor_dequant = (tensor_int - offset) * scale

        else:
            if not signed:
                abs_max = torch.max(xmax.abs(), xmin.abs()) + EPSILON
                intPart = torch.floor(torch.log2(abs_max)) + torch.ones_like(abs_max)
                fracPart = (n_bits) * torch.ones_like(intPart) - intPart
                scale = (2**fracPart).pow(-1).clamp(min=1e-6, max=1e6)

                tensor_int = torch.clamp(
                    round_ste(tensor / scale), unsigned_min, unsigned_max
                )
                tensor_dequant = tensor_int.mul(scale)
            else:
                abs_max = torch.max(xmax.abs(), xmin.abs()) + EPSILON
                intPart = torch.floor(torch.log2(abs_max)) + torch.ones_like(abs_max)
                fracPart = (n_bits - 1) * torch.ones_like(intPart) - intPart
                scale = (2**fracPart).pow(-1).clamp(min=1e-6, max=1e6)

                tensor_int = torch.clamp(
                    round_ste(tensor / scale), signed_min, signed_max
                )
                tensor_dequant = tensor_int.mul(scale)

    else:
        if offset_enabled:
            scale = ((xmax - xmin) / unsigned_max).clamp(min=1e-6, max=1e6)
            offset = round_ste(-xmin / scale)

            tensor_int = (round_ste(tensor / scale) + offset).clamp(
                unsigned_min, unsigned_max
            )

            tensor_dequant = (tensor_int - offset) * scale
        else:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (n_bits - 1) - 1)
            scale = scale.clamp(min=1e-6, max=1e6)
            tensor_int = torch.clamp(round_ste(tensor / scale), signed_min, signed_max)
            tensor_dequant = tensor_int.mul(scale)

    if group_size is not None:
        tensor_dequant = tensor_dequant.reshape(tensor_shape)

        assert group_dim_size is not None

        # reverse padding
        if pad_size:
            slices = [slice(None)] * len(tensor_shape)
            slices[axis] = slice(0, group_dim_size)
            tensor_dequant = tensor_dequant[tuple(slices)]

    assert tensor_dequant.shape == og_shape, (
        "Dequant tensor shape does not match the original shape"
    )

    return tensor_dequant


class QDQHook:
    offset_enabled: bool = False
    is_two_power: bool = True
    signed: bool = True
    axis: int = -1

    def __init__(self, name: str, n_bits: int, group_size: int | None = None, **kwargs):
        super().__init__()
        self.name = name
        self.group_size = group_size
        self.enabled = True
        self.n_bits = n_bits

        self.offset_enabled = kwargs.get("offset_enabled", self.offset_enabled)
        self.is_two_power = kwargs.get("is_two_power", self.is_two_power)
        self.signed = kwargs.get("signed", self.signed)
        self.axis = kwargs.get("axis", self.axis)

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def forward(self, module, inputs, outputs):
        if not self.enabled:
            return outputs

        assert isinstance(outputs, torch.Tensor)

        if "qkt" in self.name:
            raise NotADirectoryError
            return

        x = outputs
        x_int = qdq(x, scale=INT32_SCALE, n_bits=32, is_signed=True)
        x_qdq = qdq_group(
            x_int,
            n_bits=self.n_bits,
            group_size=self.group_size,
            offset_enabled=self.offset_enabled,
            is_two_power=self.is_two_power,
            signed=self.signed,
            axis=self.axis,
        )

        return x_qdq


class QkvQDQHook(QDQHook):
    def __init__(self, name: str, n_bits: int, group_size: int, **kwargs):
        if "axis" in kwargs:
            logger.warning(
                "`axis` is not applicable to QkvQDQHook"
                " -- user provided axis will be ignore"
            )

        super().__init__(name, n_bits, group_size, **kwargs)

    def forward(self, module, inputs, outputs):
        assert isinstance(outputs, torch.Tensor)

        if not self.enabled:
            return outputs

        x = outputs
        x_int = qdq(x, INT32_SCALE, 32, True)
        original_shape = x.shape

        q, k, v = x_int.reshape(original_shape[0], 3, -1).permute(1, 0, 2).unbind(0)

        # for Q and K we group based on head dim
        # for V we group based on seqlen

        partial_qdq_group = functools.partial(
            qdq_group,
            n_bits=self.n_bits,
            group_size=self.group_size,
            offset_enabled=self.offset_enabled,
            is_two_power=self.is_two_power,
            signed=self.signed,
        )

        q_qdq = partial_qdq_group(q, axis=-1)
        k_qdq = partial_qdq_group(k, axis=-1)
        v_qdq = partial_qdq_group(v, axis=-2)

        x_qdq = torch.cat((q_qdq, k_qdq, v_qdq), dim=-1)

        assert x_qdq.shape == original_shape, (
            "Reconstructed qkv shape should match original shape"
        )

        return x_qdq


class QktQDQHook(QDQHook):
    def __init__(self, name: str, n_bits: int, group_size: int | None, **kwargs):
        for key, value in kwargs.items():
            logger.debug(
                f"`{key}` is not applicable to QktQDQHook"
                f" -- user provided value {value} will be ignored"
            )

        if group_size is not None:
            logger.debug(
                "`group_size` is not applicable to QktQDQHook"
                f" -- user provided value will be ignored for `{name}`"
            )

        super().__init__(name, n_bits, group_size)

    def forward(self, module, inputs, outputs: torch.Tensor):
        assert isinstance(outputs, torch.Tensor)

        if not self.enabled:
            return outputs

        unsigned_min, unsigned_max = 0, (2 ** (self.n_bits)) - 1

        tensor = outputs
        tensor = qdq(tensor, INT32_SCALE, 32, True)

        EPSILON = 1.22443e-15
        axis = -1
        reduce_shape = [axis]
        xmin = tensor.amin(reduce_shape, keepdim=True)
        xmax = tensor.amax(reduce_shape, keepdim=True)
        offset = torch.zeros_like(xmax)
        # why 15?
        idx1 = xmax - xmin > 15
        idx2 = xmax - xmin <= 15
        offset[idx1] = xmax[idx1] - 15
        offset[idx2] = xmin[idx2]

        xmax = (xmax - offset).clamp(0, 15)
        xmin = (xmin - offset).clamp(0, 15)

        tensor = (tensor - offset).clamp(0, 15)

        abs_max = torch.max(xmax.abs(), xmin.abs()) + EPSILON

        intPart = torch.floor(torch.log2(abs_max)) + torch.ones_like(abs_max)
        fracPart = self.n_bits * torch.ones_like(intPart) - intPart

        scale = (2**fracPart).pow(-1)

        tensor_quant = round_ste(tensor / scale).clamp(unsigned_min, unsigned_max)
        tensor_dequant = tensor_quant * scale

        return tensor_dequant


patterns = [
    ("qkv", QkvQDQHook, {}),
    ("qkt", QktQDQHook, {}),
    ("softmax", QDQHook, {"signed": False}),
    ("v_proj", QDQHook, {"axis": -2}),
]


def parse_name_patterns(name: str) -> tuple[Type[QDQHook], dict[str, Any]] | None:
    for match_string, hook_class, kwargs in patterns:
        if match_string in name:
            return hook_class, kwargs

    return None


def quantize_weights(
    model: PreTrainedModel,
    meta: ArchitectureMeta,
    w_qconfig: QConfig,
    bit_allocation: dict[str, int] | None = None,
):
    visual_name = meta.vision_model_name

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if visual_name is not None and visual_name in name:
            module.weight = torch.nn.Parameter(
                qdq_group(
                    module.weight,
                    w_qconfig.vision.n_bits,
                    group_size=w_qconfig.vision.group_size,
                    offset_enabled=w_qconfig.vision.offset_enabled,
                    is_two_power=w_qconfig.vision.is_two_power,
                )
            )
            continue

        n_bits = w_qconfig.language.n_bits
        if bit_allocation is not None:
            n_bits = bit_allocation.get(name, n_bits)

        module.weight = torch.nn.Parameter(
            qdq_group(
                module.weight,
                n_bits,
                group_size=w_qconfig.language.group_size,
                offset_enabled=w_qconfig.language.offset_enabled,
                is_two_power=w_qconfig.language.is_two_power,
            )
        )


def add_qdq_hooks(
    model: PreTrainedModel,
    meta: ArchitectureMeta,
    a_qconfig: QConfig,
):
    hook_handlers = []
    quantizers = {}
    visual_name = meta.vision_model_name

    for name, module in model.named_modules():
        if not isinstance(module, QDQ_CLASSES):
            continue

        if visual_name is not None and visual_name in name:
            # 8 bit hooks

            logger.debug(f"Adding Vision QDQ hook on {name}")

            parsed_result = parse_name_patterns(name)
            if parsed_result is None:
                qdq_class, qdq_kwargs = QDQHook, {}
            else:
                qdq_class, qdq_kwargs = parsed_result

            quantizer = qdq_class(
                name,
                n_bits=a_qconfig.vision.n_bits,
                group_size=a_qconfig.vision.group_size,
                **qdq_kwargs,
            )
            hook_handlers.append(module.register_forward_hook(quantizer.forward))
            quantizers[name] = quantizers

        else:
            logger.debug(f"Adding Language QDQ Hook on {name}")

            parsed_result = parse_name_patterns(name)
            if parsed_result is None:
                qdq_class, qdq_kwargs = QDQHook, {}
            else:
                qdq_class, qdq_kwargs = parsed_result

            quantizer = qdq_class(
                name,
                n_bits=a_qconfig.language.n_bits,
                group_size=a_qconfig.language.group_size,
                **qdq_kwargs,
            )
            hook_handlers.append(module.register_forward_hook(quantizer.forward))
            quantizers[name] = quantizers

    return quantizers, hook_handlers
