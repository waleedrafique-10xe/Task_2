from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from GenAIQuant.logger import logger

if TYPE_CHECKING:
    from pathlib import Path

    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer


class EvaluationAPI(ABC):
    _json_file_name: str
    _valid_task_kwargs: tuple = ()

    """
    Class that evaluates the model on a particular task or a list of tasks.
    Tasks could be to measure the accuracy, perplexity or anyother metrics of the model.
    """

    def __init__(self, tasks: dict[str, Any]):
        """initialize the parameters for the evaluation class
        args:
            model: model to be evaluated for the task
            task: name of task or list of tasks to be evaluated. dict for vlmevalkit
                  tasks
            num_samples: number of samples to be evaluated for the task
            batchsize: batch size for the dataset
            seqlength: maximum sequence length
            device: device to run evaluation
        """
        self.tasks = tasks

    @abstractmethod
    def evaluate(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizer,
        output_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        """
        abstract method that evaluates the model on a particular task or list of tasks
        """

        raise NotImplementedError("Subclasses must implement evaluate()")

    def _save_results(self, results, output_dir: Path):
        file_name = output_dir / self._json_file_name
        stored_results = {}
        if os.path.isfile(file_name):
            try:
                with open(file_name, "r") as f:
                    stored_results = json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    f"{file_name} is corrupted -- This file will be over-written "
                )
                stored_results = {}
        json_to_save = {**results, **stored_results}

        with open(file_name, "w") as f:
            json.dump(json_to_save, f)

    def _validate_task_kwargs(self, task_kwargs: dict) -> bool:
        for key in task_kwargs:
            if key not in self._valid_task_kwargs:
                return False

        return True
