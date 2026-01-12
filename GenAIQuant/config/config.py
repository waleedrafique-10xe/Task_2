"""Configuration classes and enums for GenAIQuant quantization pipeline.

This module defines the configuration structure for the GenAIQuant quantization
system, including pipeline modules, algorithms, and evaluation settings.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .algorithm_configs import (
    BitAllocationConfig,
    OutlierReductionConfig,
    PtqConfig,
    QatConfig,
)
from .constants import PipelineModules


class DatasetType(str, Enum):
    """Enumeration of supported dataset types for evaluation.

    WARNING:  This is currently depending on how evaluation and datasets
        are handled. Might be removed later one
    """

    # !Note: This is likely to change with implementation of datasets module

    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class QConfigOptions(BaseModel):
    n_bits: int = 8
    group_size: int | None = None
    offset_enabled: bool = False
    is_two_power: bool = True


class QConfig(BaseModel):
    vision: QConfigOptions
    language: QConfigOptions


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation settings."""

    tasks: dict[str, Any]
    device_map: str = "auto"
    weight_only: bool = False
    generation: dict[str, Any] | None = None
    no_quantization: bool = False
    fallback_generation: dict[str, Any] | None = None


class QuantizationConfig(BaseModel):
    """Configuration for the quantization pipeline.

    This class combines all quantization-related configurations including
    pipeline modules, outlier reduction, bit allocation, PTQ, and QAT settings.

    Attributes:
        pipeline: List of pipeline modules to execute in order
        outlier_reduction: Configuration for outlier reduction algorithms
        bit_allocation: Configuration for bit allocation algorithms
        ptq: Configuration for Post-Training Quantization
        qat: Configuration for Quantization-Aware Training
    """

    pipeline: list[PipelineModules] = Field(
        default_factory=lambda: [
            PipelineModules.OUTLIER_REDUCTION,
            PipelineModules.BIT_ALLOCATION,
            PipelineModules.QAT,
        ],
        description="The list of modules to be adding to QuantizationPipeline",
    )
    outlier_reduction: OutlierReductionConfig = Field(
        default_factory=OutlierReductionConfig.default,
        description="The configuration for Outlier Reduction Modules",
    )
    bit_allocation: BitAllocationConfig = Field(
        default_factory=BitAllocationConfig.default,
        description="The configuration for Bit Allocation Modules",
    )
    ptq: PtqConfig = Field(
        default_factory=PtqConfig.default,
        description="The configuration for Post-Training Quantization",
    )
    qat: QatConfig = Field(
        default_factory=QatConfig.default,
        description="The configuration for Quantization-Aware Training",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def default(cls):
        """Create a default QuantizationConfig instance.

        Returns:
            QuantizationConfig: A new instance with default settings
        """
        return cls()


def default_activation_qconfig() -> QConfig:
    return QConfig(
        vision=QConfigOptions(
            n_bits=8,
            group_size=64,
            is_two_power=True,
            offset_enabled=False,
        ),
        language=QConfigOptions(
            n_bits=8,
            group_size=64,
            is_two_power=True,
            offset_enabled=False,
        ),
    )


def default_weight_qconfig() -> QConfig:
    return QConfig(
        vision=QConfigOptions(
            n_bits=8,
            group_size=None,
            is_two_power=False,
            offset_enabled=False,
        ),
        language=QConfigOptions(
            n_bits=4,
            group_size=64,
            is_two_power=False,
            offset_enabled=True,
        ),
    )


class Config(BaseModel):
    """Main configuration class for the GenAIQuant system.

    This is the root configuration class that combines all other configuration
    components including output settings, quantization pipeline, and evaluation.

    Attributes:
        output_dir: Directory path for saving outputs (default: "output")
        quantization: Configuration for the quantization pipeline
        evaluation: Configuration for model evaluation (required)
    """

    output_dir: Path = Field(default=Path("output"))
    device: str = Field(default="cuda:0")

    a_qconfig: QConfig = Field(default_factory=default_activation_qconfig)
    w_qconfig: QConfig = Field(default_factory=default_weight_qconfig)

    fuse_layernorms: bool = Field(
        default=True,
        description="Whether to fuse layer norms into following Conv or Linear layers",
    )
    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig.default,
        description="The configuration for setting up the QuantizationPipeline",
    )

    evaluation: EvaluationConfig | None = Field(
        None, description="Configuration for evaluation"
    )

    @classmethod
    def from_dict(cls, data: dict):
        """Create a Config instance from a dictionary.

        Args:
            data: Dictionary containing configuration values

        Returns:
            Config: A new Config instance initialized with the provided data
        """
        return cls(**data)

    @staticmethod
    def from_file(path: Path | str) -> "Config":
        if isinstance(path, str):
            path = Path(path)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"No Config file found at {path.as_posix()}")

        # Determine file type based on extension
        ext = path.suffix.lower()

        with open(path, "r") as f:
            if ext == ".json":
                data = json.loads(f.read())
            elif ext in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Unsupported file type: {ext}. Use .json, .yaml, or .yml"
                )

        return Config.from_dict(data)
