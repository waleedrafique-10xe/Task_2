from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from build_dataset import build_dataset_from_config
from transformers.processing_utils import ProcessorMixin
from vlmeval.inference import infer_data_job
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.inference_video import infer_data_job_video
from vlmeval.vlm.custom_model import CustomVLM

from GenAIQuant.logger import logger
from GenAIQuant.utils import CACHE_DIR

from ...utils import print_results_table
from ..base import EvaluationAPI

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

torch.set_float32_matmul_precision("high")


def is_single_nested(d):
    for v in d.values():
        if isinstance(v, dict):
            return False

    return True


def remove_numpy(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            data = value.tolist()
            if len(data) == 1:
                data = data[0]
            d[key] = data
        elif isinstance(value, (np.generic,)):
            d[key] = value.item()
        elif isinstance(value, dict):
            d[key] = remove_numpy(value)
    return d


class VLMEval(EvaluationAPI):
    """
    Visual-Language Model (VLM) evaluation class inspired from actual VLMEval ToolKit
    for handling multi-modal datasets and producing benchmark results on pre-loaded
    model

    This class extends the generic `Evaluation` base class to evaluate Visual-Language
    Models (VLMs) on multiple datasets defined in a task configuration.
    It supports various data modalities (e.g., images, videos, multi-turn text)
    and generates formatted result files along with human-readable evaluation reports.

    Key Features:
        - Loads and processes benchmark datasets using predefined configurations.
        - Supports multi-turn datasets and video-based evaluations.
        - Generates results in `.xlsx` or `.tsv` formats, depending on dataset type.
        - Creates task-specific reports and, when applicable, prepares results for
          official leaderboard submissions.
    """

    _json_file_name = "vlm_eval.json"
    _valid_task_kwargs = ("num_samples", "dataset", "class")

    def __init__(
        self,
        tasks: dict[str, Any],
        generation_config: dict[str, Any] | None = None,
        fallback_generation_config: dict[str, Any] | None = None,
    ):
        """
        Initialization of the LMEval class
        """
        super().__init__(tasks)
        self.generation_config = generation_config or {}
        self.fallback_generation_config = fallback_generation_config or {}

    def _sanitize_results(self, dataset_name, results):
        results_json = None
        if isinstance(results, pd.DataFrame):
            if "split" in results.columns:
                results_json = results.set_index("split").T.to_dict()
            else:
                results.index = results.index.astype(str)
                results_json = results.T.to_dict()

        if results_json is not None:
            results = results_json

        remove_numpy(results)

        if is_single_nested(results):
            results = {dataset_name: results}

        return results

    def evaluate(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizer | PreTrainedTokenizerFast,
        output_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        """
        Evaluate a vision-language model across multiple VLMEvalKit benchmark datasets.

        This method wraps the provided model in a CustomVLM interface, runs inference
        on each configured dataset, and collects evaluation metrics. Results are cached
        and optionally saved to disk.

        Args:
            model: The pre-trained model to evaluate (typically a vision-language model).
            processor: The processor/tokenizer associated with the model. Must be a
                ProcessorMixin for vision-language tasks.
            output_dir: Optional directory where evaluation results will be saved.
                If None, results are only cached in the default cache directory.

        Returns:
            A nested dictionary mapping dataset names to their evaluation metrics.
            Structure: {dataset_name: {metric_name: metric_value, ...}, ...}

        Workflow:
            1. Wraps the model with CustomVLM for VLMEvalKit compatibility.
            2. Iterates through each dataset configured in self.tasks.
            3. Builds the dataset using VLMEvalKit's dataset builder.
            4. Determines output format based on dataset type (.xlsx or .tsv).
            5. Runs inference using the appropriate handler:
               - infer_data_job_video: for video modality datasets
               - infer_data_job_mt: for multi-turn (MT) conversational datasets
               - infer_data_job: for standard image-text datasets
            6. Evaluates predictions using dataset-specific evaluation methods.
            7. Sanitizes and aggregates results across all datasets.
            8. Saves results to output_dir if provided.
            9. Prints a summary table of all results.

        Note:
            - All intermediate results are cached in vlmevalkit_cache subdirectory.
            - Model name is hardcoded as "GenAIQuant" for cache file naming.
            - The processor must be a ProcessorMixin instance (assertion enforced).
        """
        assert isinstance(processor, ProcessorMixin)

        vlm_model = CustomVLM(
            model,
            processor,
            generation_config=self.generation_config,
            fallback_generation_config=self.fallback_generation_config,
        )
        # TODO: This is a temporary fix to enable GPU on vlm-evalkit runs
        # vlm_model.model.to("cuda")

        results = {}

        result_cache_dir = Path(CACHE_DIR)

        model_name = "GenAIQuant"  # kept for vlmevalkit compatibility purposes

        if output_dir is not None:
            result_cache_dir = output_dir

        result_cache_dir = result_cache_dir / "vlmevalkit_cache"
        os.makedirs(result_cache_dir, exist_ok=True)

        for dataset_name, dataset_config in self.tasks.items():
            if not self._validate_task_kwargs(dataset_config):
                logger.warning(
                    f"Invalid task kwargs. Valid keys are {dataset_config.keys()}"
                    f"\n\t  -- {dataset_name} will be skipped"
                )
                continue

            dataset = build_dataset_from_config(self.tasks, dataset_name)

            if "num_samples" in dataset_config:
                dataset.data = dataset.data.head(dataset_config["num_samples"])

            if dataset is None:
                logger.error(
                    f"Empty dataset: {dataset_name} -- This task will be skipped"
                )
                continue

            result_file_base = f"{model_name}_{dataset_name}.xlsx"
            result_file = result_cache_dir / result_file_base

            # Handling Multi-Turn Dataset
            if dataset.TYPE == "MT":
                result_file_base = result_file_base.replace(".xlsx", ".tsv")

            # Perform the Inference
            if dataset.MODALITY == "VIDEO":
                vlm_model = infer_data_job_video(
                    vlm_model,
                    work_dir=result_cache_dir,
                    model_name=model_name,
                    dataset=dataset,
                    result_file_name=result_file,
                    verbose=True,
                    use_vllm=False,
                )
            elif dataset.TYPE == "MT":
                vlm_model = infer_data_job_mt(
                    vlm_model,
                    work_dir=result_cache_dir,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=True,
                    use_vllm=False,
                )
            else:
                vlm_model = infer_data_job(
                    vlm_model,
                    work_dir=result_cache_dir,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=True,
                    use_vllm=False,
                )

            assert result_file is not None
            eval_results = dataset.evaluate(result_file.as_posix())
            sanitized_results = self._sanitize_results(dataset_name, eval_results)

            results[dataset_name] = sanitized_results

            if output_dir is not None:
                logger.info(
                    f"Saving Results of {dataset_name} to {output_dir.as_posix()}"
                )
                self._save_results(results, output_dir)

        print_results_table(results)

        return results
