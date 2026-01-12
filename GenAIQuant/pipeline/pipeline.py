from typing import List, Type

from typing_extensions import TYPE_CHECKING

from ..algorithms import bit_allocation, outlier_reduction, ptq, qat
from ..algorithms.base import Algorithm
from ..config.constants import PipelineModules
from ..interfaces.base import ModuleInterface

if TYPE_CHECKING:
    from ..config import Config


class PipelineBuilder:
    def __init__(self) -> None:
        self.pipeline: List[Type[Algorithm]] = []
        self.config: Config | None = None

    def add(self, algorithm: str | Type[Algorithm]) -> "PipelineBuilder":
        if not (
            isinstance(algorithm, str)
            or (isinstance(algorithm, type) and issubclass(algorithm, Algorithm))
        ):
            raise TypeError(
                "Algorithm must be of type str, or a subclass of Algorithm."
            )

        algorithm_class = None
        if isinstance(algorithm, str):
            modules = [outlier_reduction, bit_allocation, ptq, qat]

            for module in modules:
                if algorithm in module.__all__:
                    algorithm_class = getattr(module, algorithm)
                    break
            if algorithm_class is None:
                raise ValueError(f"Algorithm `{algorithm}` is not available.")
        else:
            algorithm_class = algorithm

        self.pipeline.append(algorithm_class)
        return self

    def build(self) -> "Pipeline":
        return Pipeline(self.pipeline)

    def build_from_config(self, config: "Config") -> "Pipeline":
        self.config = config

        modules_list = config.quantization.pipeline
        if PipelineModules.OUTLIER_REDUCTION in modules_list:
            self.pipeline.append(
                config.quantization.outlier_reduction.algorithm(config)
            )

        if PipelineModules.BIT_ALLOCATION in modules_list:
            self.pipeline.append(config.quantization.bit_allocation.algorithm(config))

        if PipelineModules.PTQ in modules_list:
            self.pipeline.append(config.quantization.ptq.algorithm(config))

        if PipelineModules.QAT in modules_list:
            self.pipeline.append(config.quantization.qat.algorithm(config))

        return Pipeline(self.pipeline)


class Pipeline:
    def __init__(self, steps: List[Algorithm] = []) -> None:
        self.pipeline: List[Algorithm] = steps
        self.interface: ModuleInterface | None = None

    def run(self, interface: ModuleInterface, **kwargs) -> ModuleInterface:
        self.interface = interface
        assert self.interface is not None

        for algorithm in self.pipeline:
            self.interface = algorithm.forward(self.interface)
        return self.interface
