from torch import nn


class BasePassThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PassThrough(BasePassThrough):
    pass


class MatMulPassThrough(BasePassThrough):
    pass


class SoftmaxPassThrough(BasePassThrough):
    pass


__all__ = ["MatMulPassThrough", "PassThrough", "SoftmaxPassThrough"]
