from typing import TYPE_CHECKING

from ..base import Algorithm

if TYPE_CHECKING:
    from ...config import Config


class OutlierReduction(Algorithm):
    def __init__(self, config: "Config"):
        super().__init__()
        self.config = config.quantization.outlier_reduction

    def forward(self, *args, **kwargs):
        raise NotImplementedError
