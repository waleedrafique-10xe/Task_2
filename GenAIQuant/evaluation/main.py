from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from GenAIQuant.algorithms.outlier_reduction.quarot.quarot import handle_qwen_patch_embed
from GenAIQuant.algorithms.outlier_reduction.rotaters import (
    DownProjHook,
    OfflineRotation,
    OutProjHook,
)
from GenAIQuant.config import Config, PipelineModules
from GenAIQuant.interfaces.base import RotationMatrices
from GenAIQuant.logger import logger
from GenAIQuant.model_preparer.model_name_mapping import SUPPORTED_ARCHITECTURES
from GenAIQuant.model_preparer.utils import ModelType
from GenAIQuant.utils import get_nested_attr, print_config

from .evaluator import Evaluator
from .models import QDQ_MODEL_TYPE_MAPPING
from .qdq import add_qdq_hooks, quantize_weights

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from GenAIQuant.config.algorithm_configs import OutlierReductionConfig
    from GenAIQuant.model_preparer.metas.meta import ArchitectureMeta


def register_rotation_hooks(
    model: PreTrainedModel,
    meta: ArchitectureMeta,
    config: OutlierReductionConfig,
    rotation_matrices: RotationMatrices,
):
    model_type = model.config.model_type
    hook_handlers = []
    # handle
    if model_type == "qwen2_5_vl":
        vision_hidden_size = model.config.vision_config.hidden_size
        patch_embed_name = "model.visual.patch_embed"
        patch_embed: nn.Module = get_nested_attr(model, patch_embed_name)

        device = next(patch_embed.parameters()).device
        assert rotation_matrices.vision_R1 is not None, "vision_R1 is None"
        handle_qwen_patch_embed(
            model, rotation_matrices.vision_R1, vision_hidden_size, device
        )

    if not config.online:
        return hook_handlers

    rotate_v_layers = meta.online_rotations["rotate_v"]
    rotate_output_layers = meta.online_rotations["rotate_output"]
    counter = 0

    hidden_size = model.config.text_config.hidden_size
    num_attention_heads = model.config.text_config.num_attention_heads

    for name, module in model.named_modules():
        assert meta.vision_model_name is not None
        if meta.vision_model_name in name:
            logger.debug("No online rotations on vision section of model")
            continue

        split = ".".join(name.rsplit(".", 2)[-2:])

        if split in rotate_output_layers:
            parent_module_name = ".".join(name.split(".")[:-1])
            curr_module_name = name.split(".")[-1]
            parent_module = get_nested_attr(model, parent_module_name)

            if counter % 2 == 0:
                setattr(
                    parent_module,
                    curr_module_name,
                    OutProjHook(module, hidden_size, num_attention_heads),
                )
            else:
                setattr(
                    parent_module,
                    curr_module_name,
                    DownProjHook(
                        module,
                        model.config.text_config.intermediate_size,
                        # TODO: Figure this out
                    ),
                )
            counter += 1

    return hook_handlers


@torch.no_grad
def quantize_model(
    model: PreTrainedModel,
    config: Config,
    rotation_matrices: RotationMatrices | None,
    bit_allocation: dict[str, int] | None,
):
    model_type = model.config.model_type
    meta = SUPPORTED_ARCHITECTURES.get(model_type, None)

    hook_handlers = []

    if meta is None:
        raise NotImplementedError(f"model: `{model_type}` not currently supported")

    if PipelineModules.OUTLIER_REDUCTION in config.quantization.pipeline:
        if rotation_matrices is None:
            raise ValueError(
                "Expected rotation matrices to not be None since outlier_reduction"
                " is in Config"
            )

        hook_handlers += register_rotation_hooks(
            model, meta, config.quantization.outlier_reduction, rotation_matrices
        )

    assert config.evaluation is not None, (
        "Evaluation Config should be set to be in this function"
    )

    if config.evaluation.weight_only or config.evaluation.no_quantization:
        logger.info("Activation quantization is disabled")
    else:
        print_config(config.a_qconfig, "Activation Q_Config")
        quantizers, hook_handlers = add_qdq_hooks(model, meta, config.a_qconfig)

    print_config(config.w_qconfig, "Weight Q_Config")
    if config.evaluation.no_quantization:
        logger.info("Weight quantization is disabled")
    else:
        logger.info("Quantizing models weights")
        quantize_weights(model, meta, config.w_qconfig, bit_allocation=bit_allocation)

    torch.cuda.empty_cache()

    return model, hook_handlers


def evaluate_full_precision_model(model_id: str, config: Config):
    assert config.evaluation is not None

    model_config = AutoConfig.from_pretrained(model_id)
    meta = SUPPORTED_ARCHITECTURES.get(model_config.model_type, None)

    if meta is None:
        raise NotImplementedError(
            f"model type {model_config.model_type} not currently supported"
        )

    model_class = meta.architecture_class

    model = model_class.from_pretrained(
        model_id,
        device_map=config.evaluation.device_map,
        torch_dtype=model_config.torch_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    processor = None
    tokenizer = None

    if meta.model_type == ModelType.VLM:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    eval_dir = config.output_dir / "float_eval"

    evaluator = Evaluator.build_from_config(config)
    evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        output_dir=eval_dir,
    )


def run_qdq_and_evaluate(model_id: str, config: Config):
    """
    Load a dequantized model, apply QDQ (Quantize-Dequantize) transformations,
    and evaluate its performance.

    This function performs the following steps:
    1. Loads a dequantized model from the configured output directory
    2. Optionally loads rotation matrices and bit allocation configurations
    3. Applies quantization to the model using the loaded configurations
    4. Sets up the appropriate tokenizer or processor based on model type
       (VLM or standard LLM)
    5. Evaluates the quantized model using the configured evaluation tasks

    Args:
        model_id (str):
            The Hugging Face model identifier used to load the tokenizer/processor.
            This should match the original model before quantization.

        config (Config):
            Configuration object containing:
                - output_dir: Directory containing the dequantized model
                    and optional bit allocation/rotation matrices
                - evaluation: Evaluation configuration including device_map
                    and tasks
                - Other quantization-related settings

    Returns:
        model:
            The quantized model after evaluation. The model is in evaluation mode
            and loaded on the device(s) specified in config.evaluation.device_map.

    Raises:
        AssertionError: If config.evaluation is None
        NotImplementedError: If the model architecture is not supported or if the model
                           type doesn't have a corresponding QDQ model class
        FileNotFoundError: If the dequantized model path doesn't exist

    Side Effects:
        - Creates a 'dequant_eval' subdirectory in the output directory
        - Writes evaluation results to the dequant_eval directory
        - Loads model weights into GPU/CPU memory based on device_map configuration

    Note:
        - The function expects a 'dequant' subdirectory in config.output_dir containing
          the pretrained model files
        - Optional files 'bit_allocation.json' and 'rotation_matrices.pt' will be loaded
          if present in the output_dir
        - For VLM (Vision-Language Models), a processor is used instead of a tokenizer
    """

    assert config.evaluation is not None

    output_dir = config.output_dir

    model_path = output_dir / "dequant"
    bit_allocation_path = output_dir / "bit_allocation.json"
    rotation_matrices_path = output_dir / "rotation_matrices.pt"

    model_config = AutoConfig.from_pretrained(model_path)
    model_type = model_config.model_type

    meta = SUPPORTED_ARCHITECTURES.get(model_config.model_type, None)

    if meta is None:
        raise NotImplementedError(
            f"model type {model_config.model_type} not currently supported"
        )

    model_class = QDQ_MODEL_TYPE_MAPPING.get(model_type, None)

    if model_class is None:
        raise NotImplementedError(f"Model type `{model_type}` not currently supported")

    # Load model and set to eval mode
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=model_config.torch_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map=config.evaluation.device_map,
    )

    model = model.eval()

    rotation_matrices = None
    bit_allocation: dict[str, int] | None = None

    if os.path.exists(rotation_matrices_path):
        rotation_matrices = RotationMatrices(**torch.load(rotation_matrices_path))

    if os.path.exists(bit_allocation_path):
        with open(bit_allocation_path, "r") as f:
            bit_allocation = json.load(f)

    if meta.model_type == ModelType.VLM:
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True
        )

    quantize_model(model, config, rotation_matrices, bit_allocation)

    processor = None
    tokenizer = None

    if meta.model_type == ModelType.VLM:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    eval_dir = output_dir / "dequant_eval"

    evaluator = Evaluator.build_from_config(config)
    evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        output_dir=eval_dir,
    )

    return model
    # call evaluator
