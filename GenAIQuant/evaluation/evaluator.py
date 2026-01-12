import os
from pathlib import Path
from typing import Any

from lm_eval.tasks import TaskManager
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from GenAIQuant.config import Config
from GenAIQuant.logger import logger

from .metrics.lmeval import LMEval
from .metrics.perplexity import SUPPORTED_PERPLEXITY_TASKS, Perplexity
from .metrics.sanity_check import SanityCheck
from .metrics.vlmevalkit.vlmeval.dataset import (
    SUPPORTED_DATASETS as VLM_SUPPORTED_DATASETS,
)
from .metrics.vlmevalkit.vlmevalkit import VLMEval


class Evaluator:
    def __init__(
        self,
        custom_tasks: dict[str, Any],
        lm_eval_tasks: dict[str, Any],
        vlm_eval_tasks: dict[str, Any],
        vlm_generation: dict[str, Any] | None = None,
        vlm_fallback_generation: dict[str, Any] | None = None,
    ):
        self.custom_tasks = custom_tasks
        self.lm_eval_tasks = lm_eval_tasks
        self.vlm_eval_tasks = vlm_eval_tasks
        self.vlm_generation = vlm_generation or {}
        self.vlm_fallback_generation = vlm_fallback_generation or {}

        self.lm_evaluator = LMEval(self.lm_eval_tasks) if lm_eval_tasks else None
        self.vlm_evaluator = (
            VLMEval(
                self.vlm_eval_tasks,
                generation_config=self.vlm_generation,
                fallback_generation_config=self.vlm_fallback_generation,
            )
            if vlm_eval_tasks
            else None
        )
        self.custom_evaluator = Perplexity(self.custom_tasks) if custom_tasks else None
        self.sanity_checker = SanityCheck()

    @staticmethod
    def build_from_config(config: Config):
        assert config.evaluation is not None

        evaluation_config = config.evaluation

        vlm_eval_tasks = {}
        lm_eval_tasks = {}
        custom_tasks = {}
        LMEVAL_SUPPORTED_TASKS = TaskManager().all_tasks

        for task_name, task_config in evaluation_config.tasks.items():
            if task_name in SUPPORTED_PERPLEXITY_TASKS:
                custom_tasks[task_name] = task_config

            elif task_name in VLM_SUPPORTED_DATASETS:
                vlm_eval_tasks[task_name] = task_config

            elif task_name in LMEVAL_SUPPORTED_TASKS:
                lm_eval_tasks[task_name] = task_config

            else:
                logger.warning(
                    f"Task {task_name} Not Supported with any"
                    " Evaluation Frameworks and will be Skipped"
                )

        return Evaluator(
            custom_tasks,
            lm_eval_tasks,
            vlm_eval_tasks,
            vlm_generation=evaluation_config.generation,
            vlm_fallback_generation=evaluation_config.fallback_generation,
        )

    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = None,
        processor=None,
        output_dir: Path | None = None,
    ):
        if processor is None and tokenizer is None:
            raise ValueError(
                "evaluation requires either one of `tokenizer` or `processor`"
                " -- Please supply either `tokenizer`or `processor`"
            )

        if processor is not None and tokenizer is not None:
            raise ValueError(
                "evaluation requires either one of `tokenizer` or `processor`"
                " -- Supplying both is not supported"
            )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        if self.sanity_checker is not None:
            logger.info("Running Sanity Checks")
            if processor is not None:
                self.sanity_checker.evaluate(model, processor, output_dir)
            elif tokenizer is not None:
                self.sanity_checker.evaluate(model, tokenizer, output_dir)

        if self.lm_evaluator is not None:
            logger.info(f"LM-Eval Tasks to run: {list(self.lm_eval_tasks.keys())}")
            if processor is not None:
                tokenizer = processor.tokenizer

            assert tokenizer is not None
            self.lm_evaluator.evaluate(model, tokenizer, output_dir)

        if self.custom_evaluator is not None:
            logger.info(f"Custom Tasks to run: {list(self.custom_tasks.keys())}")
            if processor is not None:
                tokenizer = processor.tokenizer
            assert tokenizer is not None
            self.custom_evaluator.evaluate(model, tokenizer, output_dir)

        if self.vlm_evaluator is not None:
            logger.info(f"VLM Evalkit tasks to run: {list(self.vlm_eval_tasks.keys())}")
            assert processor is not None
            self.vlm_evaluator.evaluate(model, processor, output_dir)
