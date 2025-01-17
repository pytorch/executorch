# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import torch.fx
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def conv_remainder(input_length, pad, dilation, weight, stride):
    """
    Returns the remainder of input_length; given the padding, dilation, stride,
    and kernel size.
    """
    return (input_length + 2 * pad - dilation * (weight - 1) - 1) % stride


class SizeAdjustConv2DPass(ExportPass):
    """
    Adjust the convolution input size to match the kernel size, padding, stride,
    and dilation parameters. Pytorch allows the input and kernel shape to not
    "match", in which case the remaining rows/columns are truncated. However,
    matching the size is a requirement in the TOSA specification. In case the
    input and kernel shape do not match, the following is done to meet the
    specification:

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

    conv2d_op = exir_ops.edge.aten.convolution.default
    slice_op = exir_ops.edge.aten.slice_copy.Tensor

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        modified_graph = False
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != self.conv2d_op:
                continue

            conv_node = cast(torch.fx.Node, node)
            input_node, weight, _, stride_hw, pad_hw, dilation_hw, _, _, _ = (
                conv_node.args
            )
            weight_shape = cast(torch.fx.Node, weight).meta["val"].shape
            input_shape = cast(torch.fx.Node, input_node).meta["val"].shape

            slice_args = []
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
                    slice_args.append(args)
            if len(slice_args) == 0:
                continue

            with graph_module.graph.inserting_before(node):
                last_node = cast(torch.fx.Node, input_node)
                for args in slice_args:
                    slice_node = create_node(graph, self.slice_op, (last_node,) + args)
                    last_node = slice_node
                conv_node.replace_input_with(cast(torch.fx.Node, input_node), last_node)
                modified_graph = True

        if modified_graph:
            graph_module = super().call(graph_module).graph_module
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, True)
