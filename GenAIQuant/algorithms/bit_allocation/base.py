from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import Algorithm

if TYPE_CHECKING:
    from ...config import BitAllocationConfig, Config
    from ...interfaces import ModuleInterface


class BitAllocation(Algorithm):
    def __init__(self, config: Config):
        super().__init__()
        self.config: BitAllocationConfig = config.quantization.bit_allocation

    def forward(self, interface: ModuleInterface, **kwargs):
        raise NotImplementedError
