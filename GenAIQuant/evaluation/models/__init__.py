"""This Module contains modifying modeling files for various hugging face models

Each file is named `modeling_[model type].py` and is sources directly from hugging
face transformers. Each modeling_ file is modified to allow for adding QDQ hooks on
activations

All source files picked from hugging face version

This allows us to simulate both activation and weight quantization to get an accurate
picture of quantized model inference

The files in this module are only meant for evaluation purposes
"""

from .modeling_llama import LlamaForCausalLM
from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

QDQ_MODEL_TYPE_MAPPING = {
    "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
    "llama": LlamaForCausalLM,
}

__all__ = [
    "QDQ_MODEL_TYPE_MAPPING",
    "LlamaForCausalLM",
    "Qwen2_5_VLForConditionalGeneration",
]
