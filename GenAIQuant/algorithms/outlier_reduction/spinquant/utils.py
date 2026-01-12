import math
from typing import Any, Dict

import torch
from hadamard_transform import hadamard_transform
from GenAIQuant.logger import logger
import importlib.util

if importlib.util.find_spec("fast_hadamard_transform") is not None:
    from fast_hadamard_transform import hadamard_transform as fht
else:
    logger.warning(
        "Fast Hadamard Transform not found, switching to regular hadamard transform."
    )


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(ctx, u):
        return fht(u)

    @staticmethod
    def backward(ctx, grad):
        return fht(grad)


from GenAIQuant.algorithms.outlier_reduction.quarot.had_utils import get_hadK


class RotateModule(torch.nn.Module):
    def __init__(self, Q, device):
        super(RotateModule, self).__init__()
        self.weight = torch.nn.Parameter(Q.to(dtype=torch.float64, device=device))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


class OnlineRotation:
    @staticmethod
    def rotate_v(weight: torch.nn.Parameter, had_dim: int, Q2: torch.Tensor):
        dtype = weight.data.dtype
        W = weight.data.double()

        W = W.t()
        t_shape = W.shape
        W = W.reshape(-1, t_shape[-1] // had_dim, had_dim)
        Q2 = Q2.to(W.device)
        W = W @ Q2
        W = W.reshape(t_shape)
        W = W.t()
        weight.data = W.to(dtype=dtype)

    @staticmethod
    def rotate_o(weight: torch.nn.Parameter, had_dim: int, Q2: torch.Tensor):
        dtype = weight.data.dtype
        W = weight.data.double()

        original_shape = W.shape
        W = W.reshape(-1, original_shape[-1] // had_dim, had_dim)
        Q2 = Q2.to(W.device)
        W = W @ Q2
        W = W.reshape(original_shape)
        weight.data = W.to(dtype=dtype)

    @staticmethod
    def rotate_d(weight: torch.nn.Parameter):
        dtype = weight.data.dtype
        W = weight.data.float()

        had, K = get_hadK(W.shape[-1])
        weight.data = had_mul(W, had, K).to(dtype=dtype)


class OutProjHook(torch.nn.Module):
    def __init__(self, module, hidden_size, num_attention_heads):
        super().__init__()
        self.module = module
        self.had, self.K = get_hadK(num_attention_heads)
        self.had_dim = hidden_size // num_attention_heads

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


class DownProjHook(torch.nn.Module):
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
    def rotate_forward(weight: torch.nn.Parameter, Q: torch.Tensor):
        """
        Rotate the weight matrix forward by multiplying it with the rotation
        matrix Q.

        Args:
            weight (torch.nn.Parameter): The weight matrix to be rotated.
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
    def rotate_inverse(weight: torch.nn.Parameter, Q: torch.Tensor):
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


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
