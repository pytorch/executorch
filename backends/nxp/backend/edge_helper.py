# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx import Node
from torch.nn import Parameter


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

    def _is_dequantize(node_: Node) -> bool:
        return node_.target.__name__ in {
            "quantized_decomposed.dequantize_per_tensor.default",
            "quantized_decomposed.dequantize_per_channel.default",
        }

    return _is_dequantize(node) and node_is_static_tensor(
        node.args[0], parameters_mapping
    )
