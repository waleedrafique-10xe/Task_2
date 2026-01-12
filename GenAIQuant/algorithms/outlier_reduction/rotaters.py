import math

import torch
from hadamard_transform import hadamard_transform
import importlib.util
from .spinquant.utils import HadamardTransform
from torch import nn

from GenAIQuant.utils import get_nested_attr

from .quarot.had_utils import get_hadK


def had_mul(weight, had, K):
    weight = weight.contiguous()
    if K == 1:
        if importlib.util.find_spec("fast_hadamard_transform") is not None:
            weight = (
                HadamardTransform.apply(weight) / torch.tensor(weight.shape[-1]).sqrt()
            )
        else:
            assert weight.ndim <= 3, f"Expected = 3 dimensions, got {weight.ndim}"
            if weight.ndim == 3:
                for i in range(weight.numel() // (weight.shape[-1] * weight.shape[-2])):
                    weight[i] = hadamard_transform(weight[i])
            else:
                weight = hadamard_transform(weight)
        return weight
    n = weight.shape[-1]
    W = weight.view(-1, K, weight.shape[-1] // K)
    if importlib.util.find_spec("fast_hadamard_transform") is not None:
        W = HadamardTransform.apply(W) / torch.tensor(n).sqrt()
    else:
        for i in range(W.numel() // (weight.shape[-1])):
            W[i] = hadamard_transform(W[i])
    W = had.to(device=W.device, dtype=W.dtype) @ W
    return W.reshape(weight.shape)


class OnlineRotation:
    @staticmethod
    def rotate_v(weight: nn.Parameter, had_dim: int):
        dtype = weight.data.dtype
        W = weight.data.double()

        W = W.t()
        t_shape = W.shape
        W = W.reshape(-1, t_shape[-1] // had_dim, had_dim)
        for i in range(W.numel() // t_shape[-1]):
            W[i] = hadamard_transform(W[i])
        W = W.reshape(t_shape)
        W = W.t()
        weight.data = W.to(dtype=dtype)

    @staticmethod
    def rotate_output(weight: nn.Parameter):
        dtype = weight.data.dtype
        W = weight.data.double()

        had, K = get_hadK(W.shape[-1])
        weight.data = had_mul(W, had, K).to(dtype=dtype)


class OutProjHook(nn.Module):
    def __init__(self, module, head_dim, num_attention_heads):
        super().__init__()
        self.module = module
        self.had, self.K = get_hadK(num_attention_heads)
        self.had_dim = head_dim

    def forward(self, x, *args, **kwargs):
        original_shape = x.shape
        original_dtype = x.dtype
        x = x.double()
        x = x.reshape(-1, original_shape[-1] // self.had_dim, self.had_dim)

        if self.K == 1:
            x = x.transpose(1, 2)
            for i in range(x.numel() // (original_shape[-1])):
                x[i] = hadamard_transform(x[i])
            x = x.transpose(1, 2)
        else:
            self.had = self.had.to(device=x.device, dtype=x.dtype)
            x = (self.had @ x) / math.sqrt(original_shape[-1] // self.had_dim)
        return self.module(x.reshape(original_shape).to(dtype=original_dtype))


class DownProjHook(nn.Module):
    def __init__(self, module, intermediate_size):
        super().__init__()
        self.module = module
        self.had, self.K = get_hadK(intermediate_size)

    def forward(self, x, *args, **kwargs):
        return self.module(had_mul(x.double(), self.had, self.K).to(dtype=x.dtype))


class OfflineRotation:
    """
    Groups functionality relating to offline rotations into a single namespace
    for convenience and code readability
    """

    @staticmethod
    def rotate_forward(weight: nn.Parameter, Q: torch.Tensor):
        """
        Rotate the weight matrix forward by multiplying it with the rotation
        matrix Q.

        Args:
            weight (nn.Parameter): The weight matrix to be rotated.
            Q (torch.Tensor): The rotation matrix to apply.
        """

        dtype = weight.data.dtype
        device = weight.device

        assert Q.shape[0] == Q.shape[1], "Assumes Q is always a square matrice"

        repeat_dim = weight.shape[-1] // Q.shape[0]

        if repeat_dim > 1:
            rotated_weight = torch.zeros_like(weight).double()
            spatial_merge_size = 2
            # TODO: Make this code good
            for row_block in range(spatial_merge_size**2):
                for col_block in range(spatial_merge_size**2):
                    row_start = row_block * Q.shape[0]
                    row_end = row_start + Q.shape[0]
                    col_start = col_block * Q.shape[0]
                    col_end = col_start + Q.shape[0]

                    # Extract the block and apply the transformation
                    block = weight[row_start:row_end, col_start:col_end]
                    rotated_block = torch.matmul(block.double(), Q).double()

                    # Update in the rotated weight matrix
                    rotated_weight[row_start:row_end, col_start:col_end] = rotated_block

            weight.data = rotated_weight.to(device=device, dtype=dtype)
            return

        A = weight.double()
        weight.data = torch.matmul(A, Q).to(device=device, dtype=dtype)

    @staticmethod
    def rotate_inverse(weight: nn.Parameter, Q: torch.Tensor):
        """
        Rotate the weight matrix in the inverse direction by multiplying it with the
        transpose of Q.

        Args:
            weight (torch.nn.Parameter): The weight matrix to be rotated.
            Q (torch.Tensor): The rotation matrix whose transpose will be used.
        """
        dtype = weight.data.dtype
        device = weight.device

        A = weight.data.double()
        weight.data = torch.matmul(Q.T, A).to(device=device, dtype=dtype)


def remove_online_rotater_modules(model: nn.Module):
    for name, module in model.named_modules():
        if not isinstance(module, (OutProjHook, DownProjHook)):
            continue

        parent_module_name = ".".join(name.split(".")[:-1])
        curr_module_name = name.split(".")[-1]
        parent_module = get_nested_attr(model, parent_module_name)

        setattr(parent_module, curr_module_name, module.module)
