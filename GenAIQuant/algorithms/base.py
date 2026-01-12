from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..interfaces import ModuleInterface


class Algorithm(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, interface: ModuleInterface, **kwargs) -> ModuleInterface:
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__
