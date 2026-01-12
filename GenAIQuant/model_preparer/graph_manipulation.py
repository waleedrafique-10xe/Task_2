"""
Graph Manipulation Utilities for Model Preparation

This module provides graph manipulation functions for preparing models,

WARNING:
- These utilities are EXPERIMENTAL and HIGHLY VOLATILE
- NOT extensively tested across all model architectures
- May cause unexpected behavior if used incorrectly
- Designed for specific use cases in model quantization and export
"""

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.ops import aten  # type: ignore
from tqdm import tqdm
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

from GenAIQuant.logger import logger

from ..utils import get_nested_attr
from .utils import NodeTypes, _get_weight_node, normalize_name

if TYPE_CHECKING:
    from torch.export import ExportedProgram
    from torch.fx.node import Node

_VALID_FUSABLE_LAYER_NORMS = {"LlamaRMSNorm", "Qwen2RMSNorm"}


def get_expanded_dict(model, metadata):
    expanded_dict = {}
    for key, value in metadata.layernorm_fuses.items():
        if "#" in key and isinstance(value, dict):
            for i in range(len(get_nested_attr(model, key.split("#")[0][:-1]))):
                for k, v in value.items():
                    expanded_dict[key.replace("#", str(i)) + "." + k] = [
                        key.replace("#", str(i)) + "." + x for x in v
                    ]
        else:
            expanded_dict[key] = list(value)
    return expanded_dict


def fuse_gemma3_rms_linear(model, metadata):
    """
    Gemma3 is not currently supported for folding. This function exists
    to provide a placeholder for future implementation. Calling this function
    Will throw a NotImplementedError.
    """

    raise NotImplementedError("Gemma3 is not currently supported for folding.")

    expanded = get_expanded_dict(model, metadata)
    state_dict = model.state_dict()
    # TODO Refactor for readability and organization
    for key, value in tqdm(expanded.items(), desc="Fusing Norms"):
        RMS = state_dict[key + ".weight"].double()

        for layer in value:
            W = state_dict[layer + ".weight"]
            dtype = W.dtype
            W = W.double()

            A = 1 + RMS
            diag = torch.diag(A)
            if "post_feedforward_layernorm" in key or "post_attention_layernorm" in key:
                raise NotImplementedError("NotImplementedError")
            else:
                state_dict[layer + ".weight"] = nn.Parameter((W @ diag).to(dtype=dtype))

            if key + ".bias" in state_dict:
                if layer + ".bias" not in state_dict:
                    B = nn.Parameter(torch.zeros(layer.out_features).double())
                else:
                    B = state_dict[layer + ".bias"].double()
                RMS_B = state_dict[key + ".bias"].double()
                state_dict[layer + ".bias"] = (B + (W @ RMS_B)).to(dtype=dtype)

        RMS = state_dict[key + ".weight"]
        dtype = RMS.dtype
        RMS = RMS.double()
        state_dict[key + ".weight"] = nn.Parameter(torch.zeros_like(RMS))

        if key + ".bias" in state_dict:
            B = state_dict[key + ".bias"]
            state_dict[key + ".bias"] = torch.zeros_like(B)

    model.load_state_dict(state_dict)


def fuse_rms_linear(model, metadata):
    expanded = get_expanded_dict(model, metadata)
    state_dict = model.state_dict()
    # TODO Refactor for readability and organization
    for key, value in tqdm(expanded.items(), desc="Fusing Norms"):
        rms_weight_name = f"{key}.weight"
        rms_bias_name = f"{key}.bias"

        RMS = state_dict[key + ".weight"].double()

        rms_module = get_nested_attr(model, key)

        if isinstance(rms_module, Gemma3RMSNorm):
            logger.error("Folding `Gemma3RMSNorm` not currently supported")

        for layer in value:
            linear_weight_name = f"{layer}.weight"
            linear_bias_name = f"{layer}.bias"

            W = state_dict[linear_weight_name]
            dtype = W.dtype
            W = W.double()

            repeat_count = W.shape[-1] // RMS.shape[-1]
            if repeat_count > 1:
                # this condition checks a special case wtihin Qwen2.5 VL's patch merger
                # where you have norm => reshape => weight. To counter the reshape we
                # can repeat the weight N times to match the weight matrix

                RMS_repeat = RMS.clone().repeat(repeat_count)

                state_dict[linear_weight_name] = nn.Parameter(
                    (W * RMS_repeat).to(dtype=dtype)
                )
            else:
                state_dict[linear_weight_name] = nn.Parameter((W * RMS).to(dtype=dtype))

            if rms_bias_name in state_dict:
                B = (
                    state_dict[linear_bias_name].double()
                    if linear_bias_name in state_dict
                    else nn.Parameter(torch.zeros(W.shape[-1]).double())
                )

                RMS_B = state_dict[rms_bias_name].double()
                state_dict[linear_bias_name] = (B + (W @ RMS_B)).to(dtype=dtype)

        RMS = state_dict[rms_weight_name]
        state_dict[rms_weight_name] = nn.Parameter(torch.ones_like(RMS))

        if rms_bias_name in state_dict:
            RMS_B = state_dict[rms_bias_name]
            state_dict[rms_bias_name] = torch.zeros_like(RMS_B)

    model.load_state_dict(state_dict)


def _travese_graph_until_linear(node: "Node"):
    """
    Traverses the computation graph from a given node until a linear layer is found.

    This function recursively explores the graph by following the single user of a node,
    specifically looking for a linear layer (aten.linear.default).

    Args:
        node: A graph node to start the traversal from.

    Returns:
        list: A list containing the linear layer node if found.
        None: If no linear layer is found or the graph cannot be traversed.

    Notes:
        - Only supports graphs with a single user per node.
        - Specifically handles slice operations during traversal.
    """
    users = list(node.users)

    if node.target == aten.linear.default:
        return [node]

    if len(users) == 1 and node.target == aten.slice.Tensor:
        return _travese_graph_until_linear(users[0])

    return None


def _find_fusable_children(node: "Node") -> list["Node"] | None:
    """
    Identifies fusable linear layer children for a given multiplication node.

    This function checks if a multiplication node can be fused with a layer
    normalization operation by examining its parent module and downstream nodes.

    Args:
        node (Node): The input graph node to analyze for fusability.

    Returns:
        list[Node] | None:
            - A list of linear layer nodes if the multiplication can be fused
            - None if the node does not meet fusability criteria

    Criteria for fusability:
    1. Node must be a multiplication operation
    2. Parent module must be a valid layer normalization type
    3. All users must be linear layers or traceable to a linear layer

    Notes:
        - Checks the module hierarchy and graph structure
        - Supports recursive graph traversal to find linear layer connections
    """
    if node.op != NodeTypes.FUNCTION:
        return

    # check if node is a mul
    if node.target != aten.mul.Tensor:
        return

    # check if direct parent is a valider layernorm type
    module_stack = [value for _, value in node.meta["nn_module_stack"].values()]
    parent_module_name = module_stack[-1]
    parent_module_type = parent_module_name.split(".")[-1]

    if parent_module_type not in _VALID_FUSABLE_LAYER_NORMS:
        return None

    # check if all users are linear
    users = list(node.users)
    if all(x.target == aten.linear.default for x in users):
        return users

    # traverse graph recursvily to find if useers always returns a linear
    elif len(users) == 1:
        linear = _travese_graph_until_linear(users[0])
        return linear

    return


def fuse_rmsln_linear(xm: "ExportedProgram"):
    """
    Fuses RMSNorm (Root Mean Square Layer Normalization) with subsequent linear layers.

    This function performs the following operations:
    1. Identifies fusable layer norm nodes in the exported program's graph
    2. Merges layer norm weights with linear layer weights
    3. Removes redundant layer norm nodes from the graph
    4. Updates the state dictionary and graph signature

    The fusion process helps optimize the computational graph by:
    - Reducing the number of operations
    - Merging layer norm scaling with linear layer weights

    Args:
        xm (ExportedProgram): The exported program to modify.

    Returns:
        ExportedProgram: The modified exported program with fused layer norms.

    Notes:
        - Modifies the graph in-place
        - Supports specific layer norm types defined in _VALID_FUSABLE_LAYER_NORMS
        - Removes layer norm nodes that can be fused
    """
    weights_to_merge = {}
    keys_to_delete = set()

    # Identifies fusable layer norm nodes in the exported program's graph
    # and removes redundany nodes from graphs
    for node in xm.graph.nodes:
        fusable_children = _find_fusable_children(node)

        if fusable_children is None:
            continue

        weight, prev_node = (
            node.all_input_nodes
            if node.all_input_nodes[0].op == NodeTypes.PLACEHOLDER
            else node.all_input_nodes[::-1]
        )

        weights_to_merge[weight] = []

        assert weight is not None
        assert prev_node is not None

        for child in fusable_children:
            assert child.target == aten.linear.default, (
                "Make sure we are merging with Linear Layer"
            )

            linear_weight = next(
                x for x in child.all_input_nodes if x.op == "placeholder"
            )
            weights_to_merge[weight].append(linear_weight)

        for user in list(node.users):
            user.replace_input_with(node, prev_node)

        xm.graph.erase_node(node)
        xm.graph.erase_node(weight)

        node.replace_all_uses_with(prev_node)

    xm.graph.lint()
    xm.graph_module.recompile()

    # here we modify the state dictionary with changes to weights
    for layernorm, linears in weights_to_merge.items():
        layernorm_name = xm.graph_signature.inputs_to_parameters[layernorm.name]

        keys_to_delete.add(layernorm_name)

        layernorm_tensor = xm.state_dict[layernorm_name]

        for linear in linears:
            linear_name = xm.graph_signature.inputs_to_parameters[linear.name]

            dtype = xm.state_dict[linear_name].data.dtype

            xm.state_dict[linear_name].data = (
                xm.state_dict[linear_name].data.double() * layernorm_tensor.double()
            ).to(dtype=dtype)

    # we also update graph_signature -- this is an important step for recreating
    # a module representation for forward pass
    for name in keys_to_delete:
        spec = next(x for x in xm.graph_signature.input_specs if x.target == name)
        xm.graph_signature.input_specs.remove(spec)
        del xm.state_dict[name]

    xm.graph.eliminate_dead_code()
    return xm


def _find_embedding_placeholder(xm: "ExportedProgram") -> "None | Node":
    """
    Finds an unused placeholder node corresponding to an nn.Embedding module in the
    exported graph.

    Args:
        xm (ExportedProgram): The exported PyTorch model to search for the placeholder.

    Returns:
        Node | None: The placeholder node for the embedding weight if found,
                        otherwise None.

    Criteria for selection:
    - Must be a placeholder node
    - Must have no users (unused in the graph)
    - Must be associated with an nn.Embedding module

    Note:
        - This is typically used in graph manipulation for model preparation
        - Returns the first node matching the criteria
    """
    for node in xm.graph.nodes:
        if node.op != "placeholder":
            continue

        module = node.meta["source_fn_stack"][0][1]

        if len(node.users) == 0 and module == nn.Embedding:
            return node

    return None


def untie_word_embeddings(xm: "ExportedProgram"):
    """
    Modifies an exported PyTorch model to untie word embeddings by replacing
    the original embedding weight with a new placeholder weight.

    This function is typically used in model preparation for quantization or export,
    where you want to separate the embedding weight from its original source.

    Not seprating the embedding_weight from the original weights will cause
    problems for rotations

    Args:
        xm (ExportedProgram): The exported PyTorch model to modify.

    Process:
    1. Finds the embedding layer in the graph
    2. Locates the embedding weight placeholder
    3. Replaces the original embedding weight with the new placeholder
    4. Updates the graph signature and state dictionary accordingly

    Raises:
        AssertionError: If multiple embedding layers are found,
                        if embedding weight placeholder is not found,
                        or if the required state dict updates cannot be made.

    Note:
        - Modifies the input ExportedProgram in-place
        - Recompiles the graph module after modification
        - Updates both graph signature and state dictionary
    """
    embeddings = xm.graph.find_nodes(
        op=NodeTypes.FUNCTION, target=aten.embedding.default
    )
    assert len(embeddings) == 1, "Expecting only 1 embedding layer"

    embedding = embeddings[0]
    embedding_weight = _find_embedding_placeholder(xm)

    assert embedding_weight is not None

    old_weight = _get_weight_node(embedding.all_input_nodes, xm)
    embedding.replace_input_with(old_weight, embedding_weight)

    xm.graph.lint()
    xm.graph_module.recompile()

    spec = next(
        (
            x
            for x in xm.graph_signature.input_specs
            if x.arg.name == embedding_weight.name
        ),
        None,
    )

    assert spec is not None, "Assumes node is present in Input Specs"

    # At this point there should be a discrepency in keys of state_dict
    # and parameters list from graph_signature. We compare both, the node that is
    # missing from `xm.graph_signature.parameters` is the name of the weight tensor
    # we need to update in state_dict
    # we can do a robust check by comparing the embedding_weight.name to
    # the missing_name'normalized'

    missing_names = set(xm.state_dict.keys()) - set(xm.graph_signature.parameters)
    embedding_weight_name = next(
        (
            name
            for name in missing_names
            if embedding_weight.name == normalize_name(name)
        ),
        None,
    )

    assert embedding_weight_name is not None, (
        f"The state dict should contain `{missing_names}`"
    )

    # now that we have the correct parameter name, we need to update input specs and
    # the state_dict

    spec.target = embedding_weight_name
    # ^^ tells graph to use this weight from state_dict

    xm.state_dict[embedding_weight_name] = nn.Parameter(
        xm.state_dict[embedding_weight_name].clone()
    )
