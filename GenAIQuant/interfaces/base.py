import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torchvision.transforms.functional import torch
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from GenAIQuant.algorithms.outlier_reduction.rotaters import (
    remove_online_rotater_modules,
)
from GenAIQuant.logger import logger
from GenAIQuant.model_preparer import Model


@dataclass
class RotationMatrices:
    vision_R1: Tensor | None = None
    vision_R2: Tensor | None = None
    text_R1: Tensor | None = None
    text_R2: Tensor | None = None


class ModuleInterface(BaseModel):
    """
    This class will act as the interface between the modular algorithms
    in a typical quantization pipeline.
    """

    model: Model
    processor: ProcessorMixin | None
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None
    quant_params: dict | None = Field(
        default=None,
        description="Will contain parameters like scale and zero-point/offset.",
    )
    bit_allocation: dict[str, int] | None = Field(
        default=None, description="Will contain bit allocation scheme."
    )
    rotation_matrices: RotationMatrices | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def save(self, output_dir: Path):
        if os.path.exists(output_dir):
            logger.warning(
                f"`{output_dir}` already exists ... This will be overwritten"
            )
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        if self.rotation_matrices is not None:
            rotation_matrices_path = output_dir / "rotation_matrices.pt"
            torch.save(asdict(self.rotation_matrices), rotation_matrices_path)

            remove_online_rotater_modules(self.model.model)

        model_dir = output_dir / "dequant"
        logger.info(f"Saving Quantized model with safe_tensors to {output_dir}")
        self.model.model.save_pretrained(model_dir, safe_serialization=True)  # type: ignore

        if self.processor is not None:
            self.processor.save_pretrained(model_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(model_dir)

        if self.bit_allocation is not None:
            bit_allocation_path = output_dir / "bit_allocation.json"
            logger.info(f"Saving bit allocation config to {bit_allocation_path}")
            with open(bit_allocation_path, "w") as f:
                json.dump(self.bit_allocation, f)
