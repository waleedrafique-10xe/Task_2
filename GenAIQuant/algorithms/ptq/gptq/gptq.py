from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from GenAIQuant.logger import logger
from GenAIQuant.model_preparer.utils import ModelType
from GenAIQuant.utils import recursive_map_to_device
from GenAIQuant.utils.dataset_utils import DatasetProvider
from GenAIQuant.algorithms.datasets.dataset import DatasetUtils

from ...utils.quantizers import Quantizer
from ..base import Ptq
from .base_quantization import GPTQ_Quantizer

if TYPE_CHECKING:
    from GenAIQuant.config.config import Config
    from GenAIQuant.interfaces import ModuleInterface
    from GenAIQuant.model_preparer.model import Model


GPTQConfig = {
    "seqlen": 128,
    "split": {"wikitext": "train", "c4": "train"},
    "act_order": True,
    "static_groups": True,
    "percdamp": 0.01,
    "default_bit_width": 4,
}


class CatcherException(BaseException):
    def __init__(self):
        pass


def _extract_qwen_vl_encoder_inputs(
    model: Model,
    num_samples: int,
    dataloader,
    device: str = "cpu",
):
    inputs = []
    input_kwargs = {}

    model.gpu()

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.wrapped_module: nn.Module = module

        def forward(self, hidden_states, **kwargs):
            # assumes only 1 input
            # ensures input is on CPU to conserve VRAM
            inputs.append(hidden_states.cpu())
            # TODO: should be a generic way to do this?
            # input_kwargs["attention_mask"] = attention_mask
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    input_kwargs[key] = kwargs[key].cpu()
                else:
                    input_kwargs[key] = kwargs[key]

            raise CatcherException

    assert model.vision_layers is not None
    # assert layers is not None
    layers = model.vision_layers

    wrapped_layer = Catcher(layers[0])
    layers[0] = wrapped_layer

    # TODO: map model to GPU before passing inputs for faster run
    # TODO: retrireve original model device (from device_map?)

    with torch.no_grad():
        for batch in dataloader:
            try:
                batch = batch.to(device)
                model.model(**batch)
            except CatcherException:
                # if this exception hits, the Catcher has captured the
                # required values into `input_kwargs`
                pass

    if device != "cpu":
        torch.cuda.empty_cache()

    layers[0] = wrapped_layer.wrapped_module

    # TODO: remap model to original device

    model.cpu()
    # for x in inputs:
    #     print(x.shape)
    return torch.cat([x.unsqueeze(0) for x in inputs], dim=0), input_kwargs


def _extract_llama_decoder_inputs(
    model: Model,
    num_samples: int,  # TODO: remove num samples -- not required
    seq_len: int,
    # dataloader: list[tuple[torch.Tensor, ...]],
    dataloader,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dtype = next(iter(model.model.parameters())).dtype
    # TODO: We can declare the inputs here as a list and later call torch.cat
    # to create a torch tensor. This way, we don't need to depend on the seq_len
    # and hidden size from the model
    # Need to be careful that extracted inputs are all of the same exact shape
    inputs = torch.zeros(
        (num_samples, seq_len, model.model_config.hidden_size),
        dtype=dtype,
        device=device,
    )

    # TODO: use custom class or config to determine correct decoder layers
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.wrapped_module: nn.Module = module

        def forward(self, inp, **kwargs):
            inputs[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise CatcherException

    layers: nn.ModuleList = model.layers
    wrapped_layer = Catcher(layers[0])
    layers[0] = wrapped_layer

    original_device = model.model.device

    # this assumes that before models decoder layers,
    # there exists only and always  model.embedding. this
    # assumption may not be true for all models so couuld be
    # something to look out for
    # TODO: Make sure this problem is resolved when adding
    # new models
    model.embedding.to(device)

    # device = model.device
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting inputs"):
            try:
                # model.model(batch[0].to(device))
                # batch = batch.to(device)
                print(f'batch is {batch}')
                model.model(**batch)
            except CatcherException:
                # if this exception hits, the Catcher has captured the
                # required values in `cache`
                pass

    if device != "cpu":
        torch.cuda.empty_cache()

    # some cleanup with devices
    layers[0] = wrapped_layer.wrapped_module
    del cache["i"]
    inputs = inputs.cpu()
    model.embedding.to(original_device)  # type: ignore

    inputs = inputs.unsqueeze(1)
    # in forward pass calls, the input is expected to be of shape
    # [1, seq_len, hidden_size]. Since we handle inputs 1 by 1, in the
    # forward function we do input[i] resulting in a shape of [seq_len, hidden_size]
    # which doesn't work. Instead of changing the forward calls which would
    # lead to compatibility issues with vision models, we can reshape the
    # inputs here to [N, 1, seq_len, hidden_size]

    return inputs, cache


def _extract_qwen_vl_decoder_inputs(
    model: Model,
    num_samples: int,  # TODO: remove num samples -- not required
    dataloader,
    device: str = "cpu",
):
    inputs = []
    input_kwargs = {}
    # TODO: use custom class or config to determine correct decoder layers

    model.gpu()

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.wrapped_module: nn.Module = module

        def forward(
            self,
            hidden_states,
            # position_embeddings_global: torch.Tensor,
            # position_embeddings_local: torch.Tensor,
            **kwargs,
        ):
            inputs.append(hidden_states.cpu())
            # TODO: should be a generic way to do this?
            # TODO: verify whether position embeddings global and
            # position embeddings local are always the same for every
            # input

            # input_kwargs["position_embeddings_global"] = position_embeddings_global
            # input_kwargs["position_embeddings_local"] = position_embeddings_local

            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    input_kwargs[key] = kwargs[key].cpu()
                else:
                    input_kwargs[key] = kwargs[key]

            raise CatcherException

    layers: nn.ModuleList = model.layers
    wrapped_layer = Catcher(layers[0])
    layers[0] = wrapped_layer

    original_device = model.model.device  # TODO handling of correct device

    logger.info("Running forward pass till first decoder layer to collect inputs")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting inputs"):
            try:
                batch = batch.to(device)
                model.model(**batch)
            except CatcherException:
                # if this exception hits, the Catcher has captured the
                # required values into `input_kwargs`
                pass

    if device != "cpu":
        torch.cuda.empty_cache()

    # some cleanup with devices
    layers[0] = wrapped_layer.wrapped_module

    input_kwargs.pop("past_key_values", None)

    # TODO: remap model to original device
    model.cpu()
    return torch.cat([x.unsqueeze(0) for x in inputs], dim=0), input_kwargs








class Gptq(Ptq):
    def __init__(self, config: Config):
        super().__init__(config)

        self._device = config.device
        self._output_dir = config.output_dir
        self._config = config.quantization.ptq
        self._dataset = None
        self._numsamples = self.config.numsamples
        self._wbits = self.config.wbits
        self._sym = self.config.sym
        self._groupsize = self.config.groupsize
        self._datasetname = self.config.datasetname

        self._seqlen = GPTQConfig["seqlen"]
        self._percdamp = GPTQConfig["percdamp"]
        self._act_order = GPTQConfig["act_order"]
        self._static_groups = GPTQConfig["static_groups"]

    def _create_quantizers(
        self, subset, parent_layer_name: str, bit_config: dict[str, int] | None = None
    ) -> dict[str, GPTQ_Quantizer]:
        """Create and configure GPTQ quantizers for layer components."""
        quantizers = {}

        default_bit_width = self._wbits

        for name in subset:
            full_name = f"{parent_layer_name}.{name}"

            bit_width = (
                default_bit_width
                if bit_config is None
                else bit_config.get(full_name, default_bit_width)
            )

            quantizer = GPTQ_Quantizer(subset[name])
            quantizer.quantizer = Quantizer()
            quantizer.quantizer.configure(
                bit_width, perchannel=True, sym=self._sym, mse=False
            )
            quantizers[name] = quantizer

        return quantizers

    def _collect_statistics(
        self,
        layer: nn.Module,
        subset: dict[str, nn.Module],
        quantizers: dict[str, GPTQ_Quantizer],
        inputs: torch.Tensor,
        input_kwargs: dict[str, torch.Tensor],
    ):
        def add_batch(name):
            return lambda _, inp, out: quantizers[name].add_batch(inp[0].data, out.data)

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # TODO: shouldn't need num samples here. dataset already accounts

        for j in range(self._numsamples):
            layer(inputs[j].to(self._device), **input_kwargs)[0]

        for h in handles:
            h.remove()

    @torch.no_grad()
    def process_layers(
        self,
        layers: nn.ModuleList,
        layers_name_prefix: str,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        input_kwargs: dict[str, Any],
        interface: ModuleInterface,
    ) -> dict[str, GPTQ_Quantizer]:
        quantizers = {}

        recursive_map_to_device(input_kwargs, self._device)

        for i in tqdm(range(len(layers))):
            layer_name = f"{layers_name_prefix}.{i}"
            logger.info(f"Quantizing layer: {layer_name}")
            layer = layers[i].to(self._device)

            full = self.find_layers(layer)
            sequential = [list(full.keys())]
            for names in sequential:
                subset = {n: full[n] for n in names}

                layer_quantizers = self._create_quantizers(
                    subset, layer_name, interface.bit_allocation
                )

                self._collect_statistics(
                    layer=layer,
                    subset=subset,
                    quantizers=layer_quantizers,
                    inputs=inputs,
                    input_kwargs=input_kwargs,
                )

                for name in subset:
                    logger.info(f"Quantizing sub layer: {layer_name}.{name}")

                    layer_quantizers[name].fasterquant(
                        percdamp=self._percdamp,
                        groupsize=self._groupsize,
                        actorder=self._act_order,
                        static_groups=self._static_groups,
                    )
                    quantizers["model.layers.%d.%s" % (i, name)] = layer_quantizers[
                        name
                    ].quantizer
                    layer_quantizers[name].free()

                del layer_quantizers

            for j in range(self._numsamples):
                outputs[j] = layer(inputs[j].to(self._device), **input_kwargs)[0].cpu()

            layers[i] = layer.cpu()
            inputs, outputs = outputs, inputs

            if self._device != "cpu":
                torch.cuda.empty_cache()

        recursive_map_to_device(input_kwargs, "cpu")

        return quantizers

    @torch.no_grad()
    def forward(self, interface: ModuleInterface, **kwargs):
        model = interface.model

        assert isinstance(model.model, PreTrainedModel)

        if interface.bit_allocation is not None:
            logger.info(
                "Found Bit Allocation Config in `interface`. Using"
                " `interface.bit_allocation` for GPTQ quantizers"
            )
        else:
            logger.info(
                "No Bit Allocation Config found in `interface` "
                f"using bit_width={self._wbits} for GPTQ quantizers"
            )

        if model.meta.model_type == ModelType.LLM:
            tokenizer = AutoTokenizer.from_pretrained(model.model_id)

            self._dataset = DatasetUtils.get_lm_dataset(
                tokenizer=tokenizer,
                dataset_name=self._datasetname,
                split=GPTQConfig["split"][self._datasetname],
                num_samples=self._numsamples,
                seqlen=self._seqlen,
            )
            print(f'language dataset is {self._dataset}')
        elif model.meta.model_type == ModelType.VLM:
            processor = AutoProcessor.from_pretrained(model.model_id, use_fast=True)
            self._dataset = DatasetUtils.get_vlm_dataset(
                processor,
                dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
                num_samples=self.config.numsamples,
            )
            print(f'Vision dataset is {self._dataset}')

        use_cache = model.model_config.use_cache
        model.model_config.use_cache = False

        decoder_layers = model.layers
        decoder_layers_prefix = model.meta.layers

        if model.meta.model_type == ModelType.VLM:
            assert (
                model.meta.vision_layers is not None and model.vision_layers is not None
            ), "Expecting model.vision_layers to not be None"

            vision_layers = model.vision_layers
            vision_layers_prefix = model.meta.vision_layers

            encoder_inputs, encoder_input_kwargs = _extract_qwen_vl_encoder_inputs(
                model,
                self._numsamples,
                self._dataset,
                device=self._device,
            )

            encoder_outputs = torch.zeros_like(encoder_inputs)
            quantizers = self.process_layers(
                vision_layers,
                vision_layers_prefix,
                encoder_inputs,
                encoder_outputs,
                encoder_input_kwargs,
                interface,
            )
            assert len(encoder_inputs) == self._numsamples, (
                "inputs don't align with batch size"
            )

        match model.meta.model_type:
            case ModelType.VLM:
                decoder_inputs, decoder_input_kwargs = _extract_qwen_vl_decoder_inputs(
                    model, self._numsamples, self._dataset, device=self._device
                )
                decoder_outputs = torch.zeros_like(decoder_inputs)
            case ModelType.LLM:
                decoder_inputs, decoder_input_kwargs = _extract_llama_decoder_inputs(
                    model,
                    self._numsamples,
                    self._seqlen,
                    self._dataset,
                    device=self._device,
                )
                decoder_outputs = torch.zeros_like(decoder_inputs)

        quantizers = self.process_layers(
            decoder_layers,
            decoder_layers_prefix,
            decoder_inputs,
            decoder_outputs,
            decoder_input_kwargs,
            interface,
        )

        model.model_config.use_cache = use_cache

        interface.model = model
        interface.quant_params = quantizers
        return interface

    def find_layers(
        self, layer, layers=[nn.Conv2d, nn.Linear], name=""
    ) -> dict[str, nn.Module]:
        if type(layer) in layers:
            return {name: layer}
        res = {}
        for name1, child in layer.named_children():
            res.update(
                self.find_layers(
                    child,
                    layers=layers,
                    name=name + "." + name1 if name != "" else name1,
                )
            )
        return res
