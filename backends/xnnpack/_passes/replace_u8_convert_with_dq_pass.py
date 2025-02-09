# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class ReplaceU8ConvertWithDqPass(XNNPACKPass):
    """
    Support for U8 tensors in the XNNPACK delegate is done by treating U8
    tensors as asymmetric quantized U8 tensors with a zero-point of zero and
    a scale of 1. To handle convert ops from U8 to F32, conversion is replaced
    with a dequantize operation. This pass is responsible for perfoming this
    replacement.
    """

    @staticmethod
    def can_replace_to_copy_node(node: torch.fx.Node):
        """
        Returns true if the _to_copy node can be replaced with a dequantize
        operation. This is possible if the input dtype is u8, the output dtype
        is f32, and the dim order is not changed.
        """
        if node.op != "call_function" or node.target not in [
            exir_ops.edge.aten._to_copy.default,
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        ]:
            return False

        if node.kwargs.get("dtype", None) != torch.float:
            return False

        input_node = node.args[0]
        if (
            not isinstance(input_node, torch.fx.Node)
            or "val" not in input_node.meta
            or input_node.meta["val"].dtype != torch.uint8
        ):
            return False

        if node.target == exir_ops.edge.aten._to_copy.default:
            # TODO Don't don't assume channels_first?
            if node.kwargs.get("memory_format", torch.preserve_format) not in [
                torch.preserve_format,
                torch.contiguous_format,
            ]:
                return False
        elif node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default:
            default_dim_order = list(range(len(node.meta["val"].shape)))
            if node.kwargs.get("dim_order", default_dim_order) != default_dim_order:
                return False

        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        node_list = list(graph.nodes)
        for node in node_list:
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten._to_copy.default
            ):
                if not ReplaceU8ConvertWithDqPass.can_replace_to_copy_node(node):
                    continue

                with graph.inserting_before(node):
                    dq_node = graph.call_function(
                        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                        (
                            node.args[0],  # Tensor
                            1.0,  # Scale
                            0,  # Zero point
                            0,  # Qmin
                            255,  # Qmax
                            torch.uint8,  # Dtype
                        ),
                    )

                    node.replace_all_uses_with(dq_node)
                    graph.erase_node(node)

        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
