from __future__ import annotations

from logging import ERROR
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lm_eval
from lm_eval.models.huggingface import HFLM, eval_logger
from lm_eval.tasks import TaskManager
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from GenAIQuant.logger import logger

from ..utils import print_results_table
from .base import EvaluationAPI

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin


eval_logger.setLevel(ERROR)

for handler in eval_logger.handlers:
    handler.setLevel(ERROR)


class LMEval(EvaluationAPI):
    """
    Evaluation class to check the performance of model against the language model
    evaluation metrics.

    Inherits the 'EvaluationAPI' abstract class.
    """

    _json_file_name = "lm_eval.json"
    _log_samples = False
    _confirm_run_unsafe_code = True
    _cache_requests = True
    # fmt: off
    _valid_task_kwargs = (
        "num_fewshot", "batch_size", "max_batch_size", "device", "use_cache",
        "cache_requests", "rewrite_requests_cache", "delete_requests_cache", "limit",
        "samples", "bootstrap_iters", "check_integrity", "write_out", "log_samples",
        "evaluation_tracker", "system_instruction", "apply_chat_template",
        "fewshot_as_multiturn", "gen_kwargs", "task_manager", "verbosity",
        "predict_only", "random_seed", "numpy_random_seed", "torch_random_seed",
        "fewshot_random_seed", "confirm_run_unsafe_code", "metadata",
    )
    _valid_model_kwargs = ("batch_size", "max_batch_size")
    # fmt: on

    def __init__(self, tasks: dict[str, Any]):
        """
        Initialization of the LMEval class
        """
        super().__init__(tasks)

    def evaluate(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizer | PreTrainedTokenizerFast,
        output_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        """
        Evaluate the model on the specified lm-eval task.
        """
        assert isinstance(processor, (PreTrainedTokenizer, PreTrainedTokenizerFast))

        task_manager = TaskManager()
        results = {}

        for task_name, task_config in self.tasks.items():
            kwargs_are_valid = True

            task_kwargs = {}
            if task_config is not None and (
                kwargs_are_valid := self._validate_task_kwargs(task_config)
            ):
                task_kwargs = {**task_config}

            lm_eval_model_kwargs = {
                key: value
                for key, value in task_kwargs.items()
                if key in self._valid_model_kwargs
            }

            # Wrapping the model under HFLM is required step for LM_Eval
            lm_eval_model = HFLM(
                pretrained=model, tokenizer=processor, **lm_eval_model_kwargs
            )

            if not kwargs_are_valid:
                logger.warning(
                    f"Invalid task kwargs. Valid keys are {task_config.keys()}"
                    f"\n\t  -- {task_name} will be skipped"
                )
                continue

            task_kwargs["confirm_run_unsafe_code"] = self._confirm_run_unsafe_code
            task_kwargs["log_samples"] = self._log_samples
            task_kwargs["cache_requests"] = self._cache_requests

            logger.info(f"Evaluating model on {task_name}")
            evaluation_output = lm_eval.simple_evaluate(
                model=lm_eval_model,
                tasks=[task_name],
                task_manager=task_manager,
                **task_kwargs,
            )

            assert evaluation_output is not None

            results[task_name] = evaluation_output["results"]

            # save results after every task
            if output_dir is not None:
                logger.info(f"Saving Results of {task_name} to {output_dir.as_posix()}")
                self._save_results(results, output_dir)

        print_results_table(results)

        return results
