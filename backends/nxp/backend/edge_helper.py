# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

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

# A set of operators which could possibly be no-ops in certain conditions. The operators in this set will be proclaimed
#  as no-ops (and potentially not delegated), if their input and output tensors are equal (when run on random data).
no_op_candidates = {
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.sub.Tensor,
}


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


def try_get_dequantized_data(
    dequantize_node: Node, parameters_mapping: dict[str, Parameter]
) -> Parameter | None:
    """Get the dequantized data from the following pattern. The dequantization formula is `r = (q - Z) * S`, where `q`
        represents the static quantized data.

         ┌─────────────────────────┐
         │ <static_quantized_data> │
         └────────────┬────────────┘
                      │
                ┌─────▼──────┐
                │ Dequantize │
                └─────┬──────┘
                      ▼


    :param dequantize_node: The Dequantize node from the pattern, which dequantizes the static quantized data.
    :param parameters_mapping: Dict mapping tensor names to their static data. Should be inferred from the
                                `state_dict` attribute of an edge program.
    :return: The dequantized static parameter, or `None` if the data is not available.
    """
    if not _is_dequantize(dequantize_node):
        return None

    if not node_is_static_tensor(param := dequantize_node.args[0], parameters_mapping):
        return None

    # The pattern is correct. Dequantize the static data and return it.
    scale, zp = get_quantization_parameters_for(dequantize_node)
    quantized_data = parameters_mapping[param.name]

    dequantized_data = (quantized_data - zp) * scale
    return dequantized_data


def is_no_op_on_neutron(node: Node, parameters_mapping: dict[str, Parameter]) -> bool:
    """Check if a node is a no-op operation from the perspective of Neutron."""
    if node.op != "call_function":
        raise ValueError(
            f"is_no_op_on_neutron(): Expected call_function node, got {node.op}."
        )

    if node.target in [
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
        exir_ops.edge.aten.clone.default,
    ]:
        # Known operators which are always no-ops on Neutron.
        return True

    if node.target == exir_ops.edge.aten.cat.default and len(node.args[0]) == 1:
        # Concatenation with 1 input is a no-op.
        return True

    # For any other operators, run them with random data and see if the output is identical to the input.
    torch.manual_seed(42)
    # noinspection PyBroadException
    try:
        input_data = None
        args_with_random_data = []
        for arg in node.args:
            match arg:
                case Node():
                    # `arg` is either another operator, a model input, or a static parameter.

                    if (
                        data := try_get_dequantized_data(arg, parameters_mapping)
                    ) is not None:
                        # The `arg` is a static parameter. Use it's actual static data during the no-op test.
                        args_with_random_data.append(data)

                    else:
                        # The `arg` is a compute node or a model input. Replace it with random data for the no-op test.
                        if input_data is not None:
                            # Some random input data for `node` has already been stored, which means that the node has
                            #  more than 1 dynamic input node. Therefore, it cannot be a no-op.
                            return False

                        # Generate the random data. Use the range [-5, 5) to avoid proclaiming operations like Relu as
                        #  no-ops.
                        val = arg.meta["val"]
                        input_data = torch.rand(val.shape, dtype=val.dtype) * 10 - 5
                        args_with_random_data.append(input_data)

                case list():
                    # Lists of input nodes are not supported to keep the code simple. It is not crucial to support this
                    #  case as the affected operators are either not supported on Neutron, or are extremely unlikely to
                    #  be no-ops (e.g. GRU). One exception is `aten.cat`, which is explicitly supported above.
                    return False

                case _:
                    # Generic argument (value). Not an input from a previous node. Store it in the arguments for the
                    #  no-op test.
                    args_with_random_data.append(arg)

        # Run the operator with the random data. If the input equals the output, the node is considered a no-op.
        output_data = node.target(*args_with_random_data)

        val = node.meta["val"]
        if (
            output_data.dtype == val.dtype
            and output_data.shape == val.shape
            and torch.all(input_data == output_data)
        ):
            # The operator preserves the shape, data type, and data. Therefore, it is a no-op from the perspective of
            #  Neutron.
            if node.target in no_op_candidates:
                return True
            else:
                logging.info(
                    f"Found the operator `{node.target}`, which appears to be a no-op, but is not in the "
                    "known no-op list. Please report this issue."
                )
                return False

        else:
            # Type, shape, or data doesn't match.
            return False

    except Exception:
        # If execution fails, assume it's not a no-op.
        return False


def node_has_well_defined_shape(node: Node) -> bool:
    if (val := node.meta.get("val")) is None:
        # The node doesn't have a shape stored at all.
        return False

    return all(isinstance(dim, int) and dim > 0 for dim in val.shape)
