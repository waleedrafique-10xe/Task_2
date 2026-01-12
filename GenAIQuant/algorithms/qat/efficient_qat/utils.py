from math import inf
from typing import Optional

import torch
from torch import nn

from .quantizer.int_linear_fake import QuantLinear


class MultiBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_list = nn.ModuleList([])

    def add_block(self, block):
        self.block_list.append(block)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        for block in self.block_list:
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids
            )[0]
        return (hidden_states,)


def set_weight_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find("weight") > -1 and not (
            n.find("scale") > -1 or n.find("zero_point") > -1
        ):
            m.requires_grad = requires_grad
    return iter(params)


def weight_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("weight") > -1 and not (
            n.find("scale") > -1 or n.find("zero_point") > -1
        ):
            params.append(m)
    return iter(params)


def set_quant_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find("scale") > -1 or n.find("zero_point") > -1:
            m.requires_grad = requires_grad
    return iter(params)


def quant_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find("scale") > -1 or n.find("zero_point") > -1:
            params.append(m)
    return iter(params)


def trainable_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if m.requires_grad:
            params.append(m)
    return iter(params)


def trainable_parameters_num(model):
    params = []
    total = 0
    for n, m in model.named_parameters():
        if m.requires_grad:
            total += m.numel()
    return total


def set_quant_state(model, weight_quant: bool = False):
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.set_quant_state(weight_quant)


@torch.no_grad()
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight.data = module.weight_quantizer(module.weight.data)


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = (
            truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        )
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)


def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}


def set_op_by_name(layer, name, new_module):
    """
    Replaces a nested submodule inside a PyTorch module by its dot-separated name.

    This utility navigates through the hierarchy of a model using the provided string
    name (like those from model.named_modules()) and replaces the target submodule
    with the provided new module (e.g., a quantized version).

    Args:
        layer (nn.Module): Root module to operate on.
        name (str): Dot-separated path to the submodule to replace.
        new_module (nn.Module): The new module that will replace the target.
    """
    levels = name.split(".")  # set the module path into components
    if len(levels) > 1:
        mod_ = layer  # Start from theroot module
        # traverse all intermediate submodules except the last one
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[
                    int(levels[l_idx])
                ]  # If the level is an index (e.g., Sequential[0])
            else:
                mod_ = getattr(mod_, levels[l_idx])  # Otherwise, use attribute access

        # Replace the target module at the final level
        setattr(mod_, levels[-1], new_module)
    else:
        # If the name is top-level (no dots), directly set the attribute
        setattr(layer, name, new_module)


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def find_layers(layer, layers=[nn.Conv2d, nn.Linear], name="") -> dict[str, nn.Module]:
    if type(layer) in layers:
        return {name: layer}
    res = {}
    for name1, child in layer.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, device):
        self._scaler = torch.amp.GradScaler(device)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
        retain_graph=False,
    ):
        self._scaler.scale(loss.float()).backward(
            create_graph=create_graph, retain_graph=retain_graph
        )
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
