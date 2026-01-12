from __future__ import annotations

from typing import Literal, Type

from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator

from ..algorithms import bit_allocation, outlier_reduction, ptq, qat
from ..algorithms.base import Algorithm
from ..algorithms.bit_allocation.base import BitAllocation
from ..algorithms.outlier_reduction.base import OutlierReduction
from ..algorithms.outlier_reduction.quarot.qutils import RotateMode
from ..algorithms.ptq.base import Ptq
from ..algorithms.qat.base import Qat
from ..config.constants import PipelineModules


def _default_algorithm_fatory(module):
    from .constants import PipelineModules

    match module:
        case PipelineModules.OUTLIER_REDUCTION:

            def outlier_reduction_factory():
                from ..algorithms.outlier_reduction import SpinQuant

                return SpinQuant

            return outlier_reduction_factory

        case PipelineModules.QAT:

            def qat_factory():
                from ..algorithms.qat import EfficientQat

                return EfficientQat

            return qat_factory

        case PipelineModules.PTQ:

            def ptq_factory():
                from ..algorithms.ptq import Gptq

                return Gptq

            return ptq_factory

        case PipelineModules.BIT_ALLOCATION:

            def bit_allocation_factory():
                from ..algorithms.bit_allocation import ShortGpt

                return ShortGpt

            return bit_allocation_factory


class MbqConfig(BaseModel):
    """Configuration for Multimodal Block-wise QAT (MBQ) for vision towers.

    When enabled, this controls vision-tower quantization settings such as
    dataset, training hyperparameters, KD loss, and bitwidth overrides.
    """

    enabled: bool = Field(default=False, description="Enable MBQ for vision tower")
    dataset_name: str = Field(
        default="lmms-lab/COCO-Caption2017",
        description="Multimodal calibration dataset",
    )
    batch_size: int = Field(default=2, ge=1)
    epochs: int = Field(default=2, ge=0)
    wbits: int = Field(default=4, ge=1)
    group_size: int = Field(default=128, ge=0)
    real_quant: bool = Field(default=False)
    teacher_device: str | None = Field(
        default=None,
        description="Device for the teacher model copy (e.g., 'cpu', 'cuda:1'). Defaults to the main QAT device.",
    )
    early_stop: int = Field(default=0, ge=0)
    off_load_to_disk: bool = Field(default=False)
    mode: Literal["block", "kd_block", "global_kd"] = Field(
        default="block",
        description=(
            "Training mode: 'block' for feature reconstruction, "
            "'kd_block' for per-block KD using teacher logits, "
            "'global_kd' reserved for future end-to-end KD."
        ),
    )

    # KD/reconstruction options (default aligns with existing block_ap which uses MSE)
    kd_enabled: bool = Field(default=True)
    tau: float = Field(default=0.7)
    kd_weight: float = Field(default=1.0)
    vision_weight: float = Field(
        default=0.1, description="Relative weight for the vision-token KD loss"
    )
    language_weight: float = Field(
        default=1.0, description="Relative weight for the language-token KD loss"
    )
    reweight: bool = Field(
        default=True,
        description="Reweight modality losses based on token counts to balance gradients",
    )
    train_size: int | None = Field(
        default=None,
        description="Optional override for number of MBQ training samples; defaults to QAT train_size when unset",
    )
    val_size: int | None = Field(
        default=None,
        description="Optional override for number of MBQ validation samples; defaults to QAT val_size when unset",
    )
    vision_token_ids: list[int] = Field(
        default_factory=list,
        description="Token ids that correspond to image tokens for modality masking",
    )

    # Optional per-layer bitwidth overrides for the vision tower
    vision_bitwidth_overrides: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def default(cls) -> "MbqConfig":
        return cls()


class OutlierReductionConfig(BaseModel):
    """Configuration for outlier reduction algorithms.

    Attributes:
        algorithm: The outlier reduction algorithm to use (default: SpinQuant)
    """

    algorithm: Type[OutlierReduction] | str = Field(
        default_factory=_default_algorithm_fatory(PipelineModules.OUTLIER_REDUCTION),  # type: ignore
        description="The outlier reduction algorithm to use",
    )
    online: bool = Field(
        default=False,
        description="Whether or not to use online hooks (operations on activations)",
    )
    rotate_mode: RotateMode = Field(
        default=RotateMode.HADAMARD, description="What type of rotation matrix to use"
    )
    max_steps: int = Field(default=100)
    per_device_train_batch_size: int = Field(default=4)
    w_bits: int = Field(default=4)
    w_asym: bool = Field(default=False)
    w_clip: bool = Field(default=True)
    w_groupsize: int = Field(default=-1)

    a_bits: int = Field(default=4)
    a_asym: bool = Field(default=True)
    a_clip_ratio: float = Field(default=1.0)
    a_groupsize: int = Field(default=-1)

    k_bits: int = Field(default=4)
    k_asym: bool = Field(default=True)
    k_clip_ratio: float = Field(default=1.0)
    k_groupsize: int = Field(default=128)

    v_bits: int = Field(default=4)
    v_asym: bool = Field(default=True)
    v_clip_ratio: float = Field(default=1.0)

    int8_down_proj: bool = Field(default=False)
    fp32_had: bool = Field(default=False)

    @field_validator("algorithm", mode="before")
    @classmethod
    def validate_algorithm(cls, value: str | Type[Algorithm]) -> Type[OutlierReduction]:
        if not isinstance(value, str) and issubclass(value, Algorithm):
            value = value.get_name()

        if value not in outlier_reduction.__all__:
            raise ValueError(
                f"Algorithm `{value}` is not available. "
                f"Available algorithms are: {', '.join(outlier_reduction.__all__)}"
            )

        algorithm_class = getattr(outlier_reduction, value)
        return algorithm_class

    @classmethod
    def default(cls) -> "OutlierReductionConfig":
        """Create a default OutlierReductionConfig instance.

        Returns:
            OutlierReductionConfig: A new instance with default settings
        """
        return cls()


class BitAllocationConfig(BaseModel):
    """Configuration for bit allocation algorithms in quantization.

    Attributes:
        algorithm (Type[BitAllocation] | str):
            Bit allocation algorithm. Defaults to standard method.

        target_avg_bitwidth (float):
            Target average bitwidth. Defaults to 4.0.

        bitwidth_options (list[int]):
            Allowed bitwidth options. Defaults to [2, 3, 4].

        dataset_name (str):
            Dataset for calibration. Defaults to "wikitext".

        num_samples (int):
            Number of calibration samples. Defaults to 128.

        seq_len (int):
            Sequence length for samples. Defaults to 2048.

        use_jacobian (bool):
            Use Jacobian-based method. Defaults to False.

        n_prune_layers (int):
            Number of layers to reduce to 2 bits

    Raises:
        ValueError: If invalid algorithm or bitwidth is specified.
    """

    algorithm: Type[BitAllocation] | str = Field(
        default_factory=_default_algorithm_fatory(PipelineModules.BIT_ALLOCATION)  # type: ignore
    )
    target_avg_bitwidth: float = Field(default=3, ge=1)
    bitwidth_options: list[int] = Field(default=[2, 3, 4])

    dataset_name: str = Field(default="wikitext")
    num_samples: int = Field(default=128, ge=1)
    seq_len: int = Field(default=1024, ge=1)
    use_jacobian: bool = False
    n_prune_layers: int = Field(default=6, ge=0)

    @field_validator("algorithm", mode="before")
    @classmethod
    def validate_algorithm(cls, value: str | Type[Algorithm]) -> Type[BitAllocation]:
        if not isinstance(value, str) and issubclass(value, Algorithm):
            value = value.get_name()

        if value not in bit_allocation.__all__:
            raise ValueError(
                f"Algorithm `{value}` is not available. "
                f"Available algorithms are: {', '.join(bit_allocation.__all__)}"
            )

        algorithm_class = getattr(bit_allocation, value)
        return algorithm_class

    @field_validator("bitwidth_options", mode="after")
    def validate_bitwidth_options(cls, value: list[int]):
        if not len(value):
            raise ValueError("bitwidth_options can not be an empty list")

        for bitwidth in value:
            if bitwidth < 1 or bitwidth > 8:
                raise ValueError(
                    f"Invalid bitwidth option: {bitwidth} "
                    "Value must be in range: [1, 8]",
                )

        return value

    @classmethod
    def default(cls) -> "BitAllocationConfig":
        """Create a default BitAllocationConfig instance.

        Returns:
            BitAllocationConfig: A new instance with default settings
        """
        return cls()


class PtqConfig(BaseModel):
    """Configuration for Post-Training Quantization (PTQ) algorithms.

    Attributes:
        algorithm: The PTQ algorithm to use (default: Gptq)
    """

    algorithm: str | Type[Ptq] = Field(
        default_factory=_default_algorithm_fatory(PipelineModules.PTQ),  # type: ignore
        description="The PTQ algorithm to use",
    )
    numsamples: int = Field(default=128)
    wbits: int = Field(default=4)
    sym: bool = Field(default=False)
    groupsize: int = Field(default=-1)
    datasetname: str = Field(
        default="wikitext", description="Dataset to use for calibration"
    )

    @field_validator("algorithm", mode="before")
    @classmethod
    def validate_algorithm(cls, value: str | Type[Algorithm]) -> Type[Ptq]:
        if not isinstance(value, str) and issubclass(value, Algorithm):
            value = value.get_name()

        if value not in ptq.__all__:
            raise ValueError(
                f"Algorithm `{value}` is not available. "
                f"Available algorithms are: {', '.join(ptq.__all__)}"
            )

        algorithm_class = getattr(ptq, value)
        return algorithm_class

    @classmethod
    def default(cls) -> "PtqConfig":
        """Create a default PtqConfig instance.

        Returns:
            PtqConfig: A new instance with default settings
        """
        return cls()


class QatConfig(BaseModel):
    """Configuration for Quantization-Aware Training (QAT) algorithms.

    Attributes:
        algorithm: The QAT algorithm to use (default: EfficientQat)
    """

    algorithm: str | Type[Qat] = Field(
        default_factory=_default_algorithm_fatory(PipelineModules.QAT),  # type: ignore
        description="The QAT algorithm to use",
    )

    cache_dir: str = Field(default="./cache/EfficientQAT/")
    save_quant_dir: bool = Field(default=True)
    real_quant: bool = Field(default=False)
    calib_dataset: str = Field(default="wikitext")
    training_seqlen: int = Field(default=2048)
    train_size: int = Field(default=4096)
    val_size: int = Field(default=64)
    train_size_in_memory: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of calibration samples to keep in RAM at once during block QAT. "
            "Defaults to train_size (i.e., keep everything in memory)."
        ),
    )
    net: type(None) = Field(default=None)
    multimodal: bool = Field(default=False)
    batch_size: int = Field(default=2)
    epochs: int = Field(default=2)
    wbits: int = Field(default=4)
    group_size: int = Field(default=128)
    quant_lr: float = Field(default=1e-4)
    weight_lr: float = Field(default=1e-5)
    min_lr_factor: int = Field(default=20)
    clip_grad: float = Field(default=0.5)
    wd: int = Field(default=0)
    early_stop: int = Field(default=0)
    off_load_to_disk: bool = Field(default=False)
    e2e: dict = Field(default={})
    target_dtype: str | None = Field(
        default=None,
        description="Optional override for the target dtype (e.g., 'float16', 'bfloat16').",
    )

    @field_validator("algorithm", mode="before")
    @classmethod
    def validate_algorithm(cls, value: str | Type[Algorithm]) -> Type[Qat]:
        if not isinstance(value, str) and issubclass(value, Algorithm):
            value = value.get_name()

        if value not in qat.__all__:
            raise ValueError(
                f"Algorithm `{value}` is not available. "
                f"Available algorithms are: {', '.join(qat.__all__)}"
            )

        algorithm_class = getattr(qat, value)
        return algorithm_class

    @classmethod
    def default(cls) -> "QatConfig":
        """Create a default QatConfig instance.

        Returns:
            QatConfig: A new instance with default settings
        """
        return cls()

    # MBQ configuration for optional vision-tower quantization
    mbq: MbqConfig | None = Field(default=None)

    @field_validator("mbq", mode="before")
    @classmethod
    def _coerce_mbq(cls, value):
        if value in (None, {}, []):
            return None
        return value
