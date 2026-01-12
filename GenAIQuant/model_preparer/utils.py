from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.export import ExportedProgram
    from torch.fx.node import Node


class ModelType(Enum):
    LLM = 1
    VLM = 2


class NodeTypes(str, Enum):
    FUNCTION = "call_function"
    MODULE = "call_module"
    PLACEHOLDER = "placeholder"
    OUTPUT = "output"


def normalize_name(name: str) -> str:
    return "p_" + name.replace(".", "_")


def _get_weight_node(input_nodes: list["Node"], xm: "ExportedProgram") -> "Node":
    """
    Identifies the weight tensor node from a list of argument nodes.

    Args:
        input_nodes (list[Node]): A list of input nodes to search for the weight tensor.
        xm (ExportedProgram): The exported PyTorch model containing graph signature.

    Returns
        Node: The node representing the weight tensor.

    Raises:
        AssertionError: If the input list does not contain exactly two nodes.
        ValueError: If neither of the input nodes is a parameter node.

    Process:
    1. Assumes two input nodes
    2. Checks if either node is in the graph signature's inputs_to_parameters
    3. Returns the first node found to be a parameter node

    Note:
        - Used in graph manipulation for identifying weight tensors
        - Expects a specific input structure
    """
    assert len(input_nodes) == 2, "Assumes only two inputs, out of which 1 is a weight"
    a, b = input_nodes
    if a.name in xm.graph_signature.inputs_to_parameters:
        return a
    elif b.name in xm.graph_signature.inputs_to_parameters:
        return b

    raise ValueError("Neither Arguements nodes of weight tensors")


def _get_comp_nodes(input_nodes: list["Node"], xm: "ExportedProgram") -> "Node":
    nodes = []
    for node in input_nodes:
        if node.name in xm.graph_signature.inputs_to_parameters:
            continue
        nodes.append(node)
    return nodes
