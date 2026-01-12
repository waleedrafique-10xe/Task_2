from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoProcessor, AutoTokenizer

from GenAIQuant.logger import logger
from GenAIQuant.utils.dtype import parse_dtype, preferred_dtype_for_device

from ..base import Qat
from .main_block_ap import main_block
from .main_e2e import e2e_qp
from .mbq import main_vision

if TYPE_CHECKING:
    from GenAIQuant.config import Config
    from GenAIQuant.interfaces import ModuleInterface


class EfficientQat(Qat):
    def __init__(self, config: Config):
        super().__init__(config)
        self._config = config

    def _resolve_target_dtype(self, model) -> torch.dtype:
        qat_cfg = self._config.quantization.qat
        device = torch.device(self._config.device)

        override = parse_dtype(qat_cfg.target_dtype)
        if override is not None:
            return torch.float32 if device.type == "cpu" else override

        if device.type == "cpu":
            return torch.float32

        inferred = model.language_dtype
        if isinstance(inferred, torch.dtype):
            return inferred

        return preferred_dtype_for_device(device)

    def forward(self, interface: ModuleInterface, **kwargs):
        model = interface.model
        qat_cfg = self._config.quantization.qat
        resolved_dtype = self._resolve_target_dtype(model)
        qat_cfg.target_dtype = resolved_dtype
        logger.info("QAT: using target dtype %s", resolved_dtype)

        # Resolve tokenizer/processor based on model capabilities.
        processor = None
        tokenizer = None
        if model.is_multimodal:
            try:
                processor = AutoProcessor.from_pretrained(model.model_id)
            except (OSError, ValueError):
                logger.warning(
                    "Failed to load processor for multimodal model_id=%s; falling back to tokenizer.",
                    model.model_id,
                )
            if processor and hasattr(processor, "tokenizer"):
                tokenizer = processor.tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.model_id)
            if model.is_multimodal and processor is None:
                logger.warning(
                    "Multimodal model_id=%s is using tokenizer only; vision QAT may be skipped.",
                    model.model_id,
                )

        updated_model = main_block(
            interface=interface,
            tokenizer=tokenizer,
            processor=processor,
            config=self._config,
        )

        e2e_cfg_dict = qat_cfg.e2e if isinstance(qat_cfg.e2e, dict) else {}
        allow_multimodal_e2e = bool(e2e_cfg_dict.get("allow_multimodal", False))
        run_e2e = bool(e2e_cfg_dict) and (
            not qat_cfg.multimodal or allow_multimodal_e2e
        )
        if run_e2e:
            updated_model = e2e_qp(
                model=model, tokenizer=tokenizer, config=self._config
            )

        mbq_cfg = qat_cfg.mbq if hasattr(qat_cfg, "mbq") else None
        mbq_enabled = (
            bool(mbq_cfg.enabled)
            if (mbq_cfg is not None and hasattr(mbq_cfg, "enabled"))
            else False
        )
        run_mbq = mbq_cfg is not None and mbq_enabled and model.is_multimodal

        if run_mbq:
            if processor is None:
                logger.warning(
                    "Skipping multimodal block quantization for model_id=%s because processor could not be loaded.",
                    model.model_id,
                )
            else:
                updated_model = main_vision(
                    interface=interface,
                    processor=processor,
                    config=self._config,
                )
        interface.model = updated_model

        return interface
