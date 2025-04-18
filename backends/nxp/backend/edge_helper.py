# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx import Node


def input_tensor(node: Node, input_index: int) -> torch.Tensor:
    if len(node.all_input_nodes) <= input_index:
        raise IndexError

    return node.all_input_nodes[input_index].meta['val']


def output_tensor(node: Node) -> torch.Tensor:
    return node.meta['val']


def tensor_rank(tensor: torch.Tensor) -> int:
    return len(tensor.size())


def input_rank(node: Node, input_index: int) -> int:
    return tensor_rank(input_tensor(node, input_index))


def input_tensor_safe(node: Node, input_index: int) -> torch.Tensor | None:
    """ Return the input tensor of 'node' at index 'input_index', or None if the node doesn't have that input.

    :param node: Edge node to get the input tensor from.
    :param input_index: Index of the input tensor to get.
    :return: The input tensor at index 'input_index', or None.
    """

    if len(node.all_input_nodes) <= input_index:
        return None

    return input_tensor(node, input_index)
