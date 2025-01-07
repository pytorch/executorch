# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, Optional

import torch.fx
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._ops import OpOverload


def conv_remainder(input_length, pad, dilation, weight, stride):
    """
    Returns the remainder of input_length; given the padding, dilation, stride,
    and kernel size.
    """
    return (input_length + 2 * pad - dilation * (weight - 1) - 1) % stride


def insert_q_dq_pair(
    graph: torch.fx.Graph,
    anchor: torch.fx.Node,
    q_params: tuple,
):
    with graph.inserting_after(anchor):
        q = create_node(
            graph=graph,
            op_target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(),  # We add the argument last
        )
        q.meta = anchor.meta

    with graph.inserting_after(q):
        dq = create_node(
            graph=graph,
            op_target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(q,) + q_params,
        )
        dq.meta = q.meta

    anchor.replace_all_uses_with(dq)
    # We add this last so the replace all uses above does not replace the quantized
    # node's first use
    q.args = (anchor,) + q_params
    return dq


def create_node(
    graph: torch.fx.Graph,
    op_target: OpOverload,
    args: tuple = (),
    kwargs: Optional[dict] = None,
):
    return graph.create_node(
        "call_function",
        op_target,
        args=args,
        kwargs=kwargs or {},
    )


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
                    slice_node = graph.create_node(
                        "call_function", self.slice_op, (last_node,) + args
                    )
                    last_node = slice_node
                conv_node.replace_input_with(cast(torch.fx.Node, input_node), last_node)
                modified_graph = True

        if modified_graph:
            graph_module = super().call(graph_module).graph_module
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, True)
