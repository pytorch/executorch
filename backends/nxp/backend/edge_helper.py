# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import GraphModule, Node
from torch.nn import Parameter

QUANTIZE_OPERATORS = [
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
]

DEQUANTIZE_OPERATORS = [
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
]


def input_tensor(node: Node, input_index: int) -> torch.Tensor:
    if len(node.all_input_nodes) <= input_index:
        raise IndexError

    return node.all_input_nodes[input_index].meta["val"]


def output_tensor(node: Node) -> torch.Tensor:
    return node.meta["val"]


def tensor_rank(tensor: torch.Tensor) -> int:
    return len(tensor.size())


def input_rank(node: Node, input_index: int) -> int:
    return tensor_rank(input_tensor(node, input_index))


def input_tensor_safe(node: Node, input_index: int) -> torch.Tensor | None:
    """Return the input tensor of 'node' at index 'input_index', or None if the node doesn't have that input.

    :param node: Edge node to get the input tensor from.
    :param input_index: Index of the input tensor to get.
    :return: The input tensor at index 'input_index', or None.
    """

    if len(node.all_input_nodes) <= input_index:
        return None

    return input_tensor(node, input_index)


def node_is_static_tensor(node: Node, parameters_mapping: dict[str, Parameter]) -> bool:
    """Return `True` if the given `node` has static data in the `parameters_mapping` dict.
    :param node: Tensor node to check for data.
    :param parameters_mapping: Dict mapping tensor names to their static data. Should be inferred from the
                                `state_dict` attribute of an edge program.
    """
    return node.name in parameters_mapping.keys()


def node_is_effectively_static_tensor(
    node: Node, parameters_mapping: dict[str, Parameter]
) -> bool:
    """Return `True` if the given `node` has static data, or follows after a `Dequantize` node with a static input.
     In the IR, the `node` will be turned into a static quantized tensor.
    :param node: Tensor node to check for data.
    :param parameters_mapping: Dict mapping tensor names to their static data. Should be inferred from the
                                `state_dict` attribute of an edge program.
    """
    if node_is_static_tensor(node, parameters_mapping):
        return True

    return _is_dequantize(node) and node_is_static_tensor(
        node.args[0], parameters_mapping
    )


def try_get_tensor_constant_from_node(
    graph_module: GraphModule, node: Node
) -> Parameter | None:
    """Get the static data from a given node. If it doesn't have any data, return `None`."""
    if node is None or node.op != "get_attr":
        return None

    target_atoms = node.target.split(".")
    attr_itr = graph_module
    for atom in target_atoms:
        if not hasattr(attr_itr, atom):
            return None
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def _is_dequantize(node_: Node) -> bool:
    return node_.op == "call_function" and node_.target in [
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    ]


def _is_quantize(node_: Node) -> bool:
    return node_.op == "call_function" and node_.target in [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_channel.default,
    ]


def previous_non_qdq_node(node: Node, input_index: int = 0) -> Node | None:
    """Return the first node which is not a `quantize` or `dequantize`, found by traversing the graph backwards
    starting with the `node.args[input_index]`,
    """
    current_node = node.args[input_index]
    while True:
        if _is_quantize(current_node) or _is_dequantize(current_node):
            current_node = current_node.args[0]
        else:
            return current_node


Scale = list[float] | float
ZeroPoint = list[int] | int


def get_quantization_parameters_for(node: Node) -> tuple[Scale, ZeroPoint] | None:
    if "quantize" not in node.target.__name__ or len(node.args) < 3:
        return None

    return node.args[1], node.args[2]  # Scale and zero_point


def get_non_qdq_users(node: Node) -> list[Node]:
    """Return a list of nodes which consume the output of `node`, but Quantize/Dequantize nodes from QDQ clusters are
     ignored. Meaning, the list of nodes [<user_1>, ..., <user_N>] from the illustration below is returned.

    If the graph does not follow the QDQ pattern, an empty list is returned.

                │
            ┌───▼────┐
            │ `node` │
            └───┬────┘
           ┌────▼─────┐
           │ Quantize │
           └────┬─────┘
                ├─────── ... ──────┐
          ┌─────▼──────┐     ┌─────▼──────┐
          │ Dequantize │ ... │ Dequantize │
          └─────┬──────┘     └─────┬──────┘
           ┌────▼─────┐       ┌────▼─────┐
           │ <user_1> │  ...  │ <user_N> │
           └────┬─────┘       └────┬─────┘

    """

    quant_nodes = list(node.users)
    if len(quant_nodes) != 1 or quant_nodes[0].target not in [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    ]:
        return []

    dequant_nodes = list(quant_nodes[0].users)
    if any(
        dequant_node.target
        not in [
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        ]
        for dequant_node in dequant_nodes
    ):
        return []

    res = []
    for dequant_node in dequant_nodes:
        res.extend(list(dequant_node.users))

    return res


def is_channels_last_dim_order(dim_order: list[int]) -> bool:
    if len(dim_order) < 3:
        return False

    return list(dim_order) == [0] + list(range(2, len(dim_order))) + [1]


def get_non_qdq_parent(node: Node, input_index: int = 0) -> Node | None:
    """Return the node which produces the input of `node` on a given index, but Quantize/Dequantize nodes from QDQ
     clusters are ignored. Meaning, the node `parent` from the illustration below is returned.

    If the graph does not follow the QDQ pattern, `None` is returned.

                │
           ┌────▼─────┐
           │ `parent` │
           └────┬─────┘
           ┌────▼─────┐
           │ Quantize │
           └────┬─────┘
          ┌─────▼──────┐
          │ Dequantize │
          └─────┬──────┘
            ┌───▼────┐
            │ `node` │
            └───┬────┘

    """

    if not _is_dequantize(dequant_node := node.args[input_index]):
        return None

    if not _is_quantize(quant_node := dequant_node.args[0]):
        return None

    return quant_node.args[0]
