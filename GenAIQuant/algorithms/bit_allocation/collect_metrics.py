from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from GenAIQuant.logger import logger
from GenAIQuant.model_preparer.utils import ModelType

if TYPE_CHECKING:
    import torch

    from ...model_preparer import Model


class ObserverException(Exception):
    pass


class BaseObserver:
    """
    Abstract base class for observing module activations during model inference.

    Provides a framework for creating hooks that monitor module inputs and outputs
    with enable/disable functionality.

    Subclasses can implement pre_forward_hook and post_forward_hook methods.
    """

    def __init__(self, module_name: str):
        self.name = module_name
        self.observer_enabled = False

    def pre_forward_hook(self, module, inputs: tuple[torch.Tensor, ...]):
        raise NotImplementedError

    def post_forward_hook(
        self,
        module,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ):
        raise NotImplementedError

    def enable(self):
        self.observer_enabled = True
        return self

    def disable(self):
        self.observer_enabled = False
        return self


def compute_block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
) -> torch.Tensor:
    """
    Calculate the influence of a block by measuring the similarity between
    input and output hidden states.

    Args:
        input_hidden_state
        output_hidden_state
        angular: If True, use angular distance instead of cosine similarity

    Returns:
        float: Influence score (lower means less important)
    """
    d = input_hidden_state.shape[-1]
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return torch.arccos(sim) / torch.pi

    return 1 - sim


class BlockInfluenceObserver(BaseObserver):
    """
    Observer for measuring block influence during model inference.

    Computes the cumulative influence of a module by analyzing input and output tensors.
    Extends BaseObserver with block-level metric collection.

    Attributes:
        block_influence (float): Cumulative measure of module's computational
        significance
    """

    def __init__(self, module_name: str):
        super().__init__(module_name)
        self.total_block_influence = 0
        self.block_influence: float = 0
        self.sample_count = 0

    def post_forward_hook(
        self,
        module,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ):
        self.sample_count += 1

        if not self.observer_enabled:
            return

        hidden_in = inputs[0].detach()
        hidden_out = outputs[0].detach()

        self.total_block_influence += (
            compute_block_influence(hidden_in, hidden_out).mean().cpu().item()
        )
        self.block_influence = self.total_block_influence / self.sample_count


def collect_metrics(
    model: Model,
    calib_loader: list[tuple[torch.Tensor, ...]],
) -> dict[str, float]:
    """
    Collect computational influence metrics for model layers using calibration data.

    Runs a forward pass on the calibration dataset, measuring each layer's
    computational significance using BlockInfluenceObserver.

    Args:
        model (Model): Model to analyze
        calib_loader (list[tuple[torch.Tensor, ...]]): Calibration dataset

    Returns:
        dict[str, float]: Layer-wise block influence metrics
    """

    observers = {}

    # add observers
    hooks = []
    observers = {}

    logger.info("Registering Observers as forward hooks")

    for i, layer in enumerate(model.layers):
        layer_name = f"{model.meta.layers}.{i}"
        observer = BlockInfluenceObserver(module_name=layer_name).enable()
        hooks.append(layer.register_forward_hook(observer.post_forward_hook))
        observers[layer_name] = observer

    if model.meta.model_type == ModelType.VLM:
        assert model.vision_layers is not None
        for i, layer in enumerate(model.vision_layers):
            layer_name = f"{model.meta.vision_layers}.{i}"
            observer = BlockInfluenceObserver(module_name=layer_name).enable()
            hooks.append(layer.register_forward_hook(observer.post_forward_hook))
            observers[layer_name] = observer

    # model.model.to(device=device)  # type: ignore
    input_device = next(model.model.parameters()).device
    logger.info("Running forward pass on calibration dataset to collect statistics")

    with torch.no_grad():
        for batch in tqdm(calib_loader, desc="Collecting Statistics"):
            if model.meta.model_type == ModelType.LLM:
                model.model(batch[0].to(input_device))
            else:
                input_kwargs = {
                    key: value.to(device=input_device) for key, value in batch.items()
                }
                model.model(**input_kwargs)

    for hook in hooks:
        hook.remove()

    logger.info("Computing RMS on collected data")
    # model.model.to(device=original_device)  # type: ignore

    torch.cuda.empty_cache()
    metrics = {layer: observer.block_influence for layer, observer in observers.items()}

    return metrics
