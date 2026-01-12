from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from GenAIQuant.logger import logger

from ..utils import print_results_table
from .base import EvaluationAPI

if TYPE_CHECKING:
    from pathlib import Path

    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin

SUPPORTED_PERPLEXITY_TASKS = ["wikitext"]


@torch.no_grad()
def perplexity(
    model, dataset, seq_len, num_samples, device: str | torch.device = "cpu"
):
    neg_log_likelihoods = []
    for idx in tqdm(range(num_samples), desc="Evaluating perplexity"):  # type: ignore
        batch = dataset[:, idx * seq_len : (idx + 1) * seq_len].to(device)
        # Handle model with different architectures

        if hasattr(model, "model") and hasattr(model, "lm_head"):
            # For models like GPT, Llama, etc.
            outputs = model.model(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)
        else:
            # Fallback for other model architectures
            outputs = model(batch)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = dataset[:, (idx * seq_len) : ((idx + 1) * seq_len)][:, 1:].to(
            logits.device
        )

        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seq_len

        if not torch.isnan(neg_log_likelihood).item():
            neg_log_likelihoods.append(neg_log_likelihood)
        else:
            logger.warning(
                "NaN encountered in perplexity calculation"
                " -- this sample will be skipped"
            )

        # Clean up memory
        del outputs, batch, logits, shift_logits, shift_labels

    torch.cuda.empty_cache()

    # Calculate perplexity
    ppl = torch.exp(
        torch.stack(neg_log_likelihoods).sum() / (len(neg_log_likelihoods) * seq_len)
    )

    return ppl.item()


class Perplexity(EvaluationAPI):
    """
    Evaluation class to check the performance of model against the perplexity metrics.
    Inherits the 'Evaluation' abstract class.
    """

    DEFAULT_TASK = "wikitext"

    _valid_task_kwargs = ("seq_len", "batch_size", "num_samples")
    _json_file_name = "custom_eval.json"
    NUM_SAMPLES = None
    SEQ_LEN = 2048
    BATCH_SIZE = 1

    def __init__(self, tasks: dict[str, Any]):
        """
        Initialization of the Perplexity class
        """
        super().__init__(tasks)

    def evaluate(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizer,
        output_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        assert isinstance(processor, (PreTrainedTokenizer, PreTrainedTokenizerFast))

        data_device = next(model.parameters()).device

        results = {}

        for task_name, task_config in self.tasks.items():
            if not self._validate_task_kwargs(task_config):
                logger.warning(
                    f"Invalid task kwargs. Valid keys are {task_config.keys()}"
                    f"\n\t  -- {task_name} will be skipped"
                )
                continue

            logger.info(f"Evaluating model on {task_name}")

            dataset = self.load_dataset(dataset=task_name, tokenizer=processor)

            if dataset is None:
                raise ValueError(f"Dataset not found for the task: {task_name}")

            seq_len = task_config.get("seq_len", self.SEQ_LEN)
            num_samples = dataset.numel() // seq_len

            if "num_samples" in task_config:
                num_samples = min(num_samples, task_config["num_samples"])

            ppl = perplexity(model, dataset, seq_len, num_samples, data_device)
            results[task_name] = {"wikitext-2-raw-v1--test": {"perplexity": ppl}}
            torch.cuda.empty_cache()

            if output_dir is not None:
                logger.info(f"Saving Results of {task_name} to {output_dir.as_posix()}")
                self._save_results(results, output_dir)
        print_results_table(results)

        return results

    def load_dataset(self, dataset, tokenizer):
        if os.path.exists(dataset):
            # If dataset is a local path, load it directly
            dataset = load_dataset(
                dataset, data_files={"validation": dataset}, split="validation"
            )
        elif isinstance(dataset, str):
            dataset_config = {"wikitext": ("wikitext", "wikitext-2-raw-v1")}
            dataset_name, version = dataset_config[dataset]
            # If dataset is a string, assume it's a Hugging Face dataset name
            try:
                dataset = load_dataset(dataset_name, version, split="test")
            except Exception as e:
                raise ValueError(f"Failed to load dataset '{dataset}': {e}")

        text = "\n\n".join(dataset["text"])
        encodings = tokenizer(text, return_tensors="pt")
        return encodings.input_ids
