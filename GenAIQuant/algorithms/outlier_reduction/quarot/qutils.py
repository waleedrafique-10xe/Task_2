from enum import Enum

import torch

from .had_utils import get_hadK


class RotateMode(str, Enum):
    RANDOM = "random"
    HADAMARD = "hadamard"


def get_rotation_matrix(size: int, rotate_mode: RotateMode, device: str):
    if rotate_mode == RotateMode.RANDOM:
        return random_rotation_matrix(size, device)
    elif rotate_mode == RotateMode.HADAMARD:
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {rotate_mode}")


def random_rotation_matrix(size, device):
    torch.cuda.empty_cache()
    mat = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(mat)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def random_hadamard_matrix(size, device):
    q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    q = q * 2 - 1
    q = torch.diag(q)

    hadK, K = get_hadK(q.shape[-1], False)
    input = q.clone().view(-1, q.shape[-1], 1)
    output = input.clone()

    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        assert hadK is not None
        input = hadK.view(1, K, K).to(input) @ input

    q = input.view(q.shape) / torch.tensor(q.shape[-1]).sqrt()

    return q.to(device)
