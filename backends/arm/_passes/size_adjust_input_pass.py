# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast, Set, Type, TypeAlias

import torch.fx
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.rewrite_conv2d_pass import RewriteConv2dPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

Slices: TypeAlias = list[tuple[int, int, int]]

conv2d_op = exir_ops.edge.aten.convolution.default
max_pooling_op = exir_ops.edge.aten.max_pool2d.default
avg_pooling_op = exir_ops.edge.aten.avg_pool2d.default
slice_op = exir_ops.edge.aten.slice_copy.Tensor

valid_operators = [conv2d_op, max_pooling_op, avg_pooling_op]


def conv_remainder(input_length, pad, dilation, weight, stride) -> int:
    """
    Returns the remainder of input_length; given the padding, dilation, stride,
    and kernel size.
    """
    return (input_length + 2 * pad - dilation * (weight - 1) - 1) % stride


def pooling_remainder(input_size, pad, kernel_size, stride) -> int:
    """
    Returns the remainder of input_length; given the padding, stride, and
    kernel size.
    """
    return (input_size + 2 * pad - kernel_size) % stride


def get_slices_conv2d(conv_node: torch.fx.Node) -> Slices:
    slices = []

    input_node, weight, _, stride_hw, pad_hw, dilation_hw, _, _, _ = conv_node.args
    weight_shape = cast(torch.fx.Node, weight).meta["val"].shape
    input_shape = cast(torch.fx.Node, input_node).meta["val"].shape

    for stride, pad, dilation, dim in zip(
        cast(list, stride_hw),
        cast(list, pad_hw),
        cast(list, dilation_hw),
        (2, 3),
    ):
        remainder = conv_remainder(
            input_shape[dim], pad, dilation, weight_shape[dim], stride
        )
        if remainder > pad:
            adjustment = remainder - pad
            args = (dim, 0, input_shape[dim] - adjustment)
            slices.append(args)

    return slices


def get_slices_pooling(pooling_node: torch.fx.Node) -> Slices:
    slices = []

    input_node = pooling_node.args[0]
    kernel_size = pooling_node.args[1]
    stride = pooling_node.args[2]
    padding = pooling_node.args[3] if len(pooling_node.args) >= 4 else [0, 0]

    # For the loop below, padding must be a list
    if isinstance(padding, int):
        padding = [padding, padding]

    input_shape = cast(torch.fx.Node, input_node).meta["val"].shape

    for kernel_length, stride_length, pad_size, dim in zip(
        cast(list, kernel_size),
        cast(list, stride),
        cast(list, padding),
        (2, 3),
    ):
        remainder = pooling_remainder(
            input_shape[dim], pad_size, kernel_length, stride_length
        )
        if remainder > pad_size:
            adjustment = remainder - pad_size
            args = (dim, 0, input_shape[dim] - adjustment)
            slices.append(args)

    return slices


def get_slices(node: torch.fx.Node) -> Slices:
    """
    Returns the remainder of input_length; given graph Node.
    """
    if node.target == conv2d_op:
        return get_slices_conv2d(node)
    elif node.target == max_pooling_op or node.target == avg_pooling_op:
        return get_slices_pooling(node)
    else:
        raise ValueError(f"Unsupported node target, was expecting {valid_operators}")


def is_valid_operator(node: torch.fx.Node) -> bool:
    if node.target == conv2d_op:
        return True
    elif node.target == max_pooling_op:
        dilation = node.args[4] if len(node.args) >= 5 else 1
        ceil_mode = node.args[5] if len(node.args) >= 6 else False

        # Dilation should be handled first by DecomposeMaxPool2DPass
        if isinstance(dilation, int):
            if dilation > 1:
                raise ValueError(
                    "Expected max_pool2d with dilation = 1, has DecomposeMaxPool2DPass been run?"
                )
        else:
            dilation = cast(list, dilation)
            if dilation[0] > 1 or dilation[1] > 1:
                raise ValueError(
                    "Expected max_pool2d with dilation = [1, 1], has DecomposeMaxPool2DPass been run?"
                )

        # If using ceil mode for rounding, the input does not need adjusting
        return not ceil_mode
    elif node.target == avg_pooling_op:
        ceil_mode = node.args[4] if len(node.args) >= 5 else False
        count_include_pad = node.args[5] if len(node.args) >= 6 else True
        divisor_override = node.args[6] if len(node.args) >= 7 else None

        return not ceil_mode and not count_include_pad and divisor_override is None

    return False


class SizeAdjustInputPass(ArmPass):
    """
    Adjusts the input size to Conv2D and Pooling operators. PyTorch allows
    the input and kernel shape to not "match", in which case the remaining
    rows/columns are truncated. However, matching the size is a requirement
    in the TOSA specification. In case the input and kernel shape do not
    match, the following is performed to meet the specification:

      1) The padding is truncated (done in the node visitor)
      2) (if neccessary) The input is truncated (done in this pass)."

    A simple example would be a 2x2 kernel (no padding, stride=2) and a 5x5
    input:

    ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐
    │ X │ X │   │   │   │    │   │   │ X │ X │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │ X │ X │   │   │   │    │   │   │ X │ X │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │ -> │   │   │   │   │   │ -> │ X │ X │   │   │   │ ->
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │    │   │   │   │   │   │    │ X │ X │   │   │   │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │    │   │   │   │   │   │    │   │   │   │   │   │
    └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘
         First pass               second pass              third pass

    ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐
    │   │   │   │   │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │ X │ X │   │ -> │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │ X │ X │   │    │   │   │   │   │ - │
    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │    │ - │ - │ - │ - │ - │
    └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘
         Fourth pass            Unvisited cells

    Cells that are never visited are marked with `-` and are never considered
    when the kernel traverses over the input, hence they can be removed.

    To match the shape of the kernel (and all parameters) with the input, a
    slice op is inserted to remove the remaining edges (rows and columns) of the
    input.
    """

    _passes_required_after: Set[Type[ExportPass]] = {RewriteConv2dPass}

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified_graph = False
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if not is_valid_operator(node):
                continue

            target_node = cast(torch.fx.Node, node)
            slice_args = get_slices(target_node)

            if len(slice_args) == 0:
                continue

            parent_node = node.args[0]
            with graph_module.graph.inserting_before(node):
                last_node = cast(torch.fx.Node, parent_node)
                for args in slice_args:
                    slice_node = create_node(graph, slice_op, (last_node,) + args)
                    last_node = slice_node
                node.replace_input_with(cast(torch.fx.Node, parent_node), last_node)
                modified_graph = True

        if modified_graph:
            graph_module = super().call(graph_module).graph_module
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, True)
