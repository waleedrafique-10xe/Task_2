from __future__ import annotations

import torch

from .config import Config
from .evaluation import run_qdq_and_evaluate
from .interfaces.base import ModuleInterface
from .logger import logger
from .model_preparer import ModelPreparer
from .model_preparer.utils import ModelType
from .pipeline.pipeline import PipelineBuilder


def quantization_pipeline_process(model_id: str, config: Config):
    model = ModelPreparer(model_id=model_id, config=config).prepare()

    processor = None
    tokenizer = None

    if model.meta.model_type == ModelType.VLM:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)

    interface = ModuleInterface(model=model, processor=processor, tokenizer=tokenizer)
    pipeline = PipelineBuilder().build_from_config(config)

    pipeline.run(interface=interface)

    interface.save(config.output_dir)
    logger.info(
        "Max GPU memory allocated:"
        f" {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )


def evaluation_pipeline_process(model_id: str, config: Config):
    assert config.evaluation is not None

    run_qdq_and_evaluate(model_id, config)
    logger.info(
        "Max GPU memory allocated:"
        f" {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
