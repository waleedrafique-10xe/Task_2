from enum import Enum


class PipelineModules(str, Enum):
    """Enumeration of available quantization pipeline modules."""

    OUTLIER_REDUCTION = "outlier_reduction"
    BIT_ALLOCATION = "bit_allocation"
    PTQ = "ptq"
    QAT = "qat"
