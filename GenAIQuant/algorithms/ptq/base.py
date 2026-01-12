from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import Algorithm

if TYPE_CHECKING:
    from GenAIQuant.config import Config
    from GenAIQuant.interfaces import ModuleInterface


class Ptq(Algorithm):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config.quantization.ptq

    def forward(self, interface: ModuleInterface, **kwargs):
        raise NotImplementedError
