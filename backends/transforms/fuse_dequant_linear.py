# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class FuseDequantLinearPass(ExportPass):
    """
    Fuses weight dequantize_per_channel nodes with linear nodes into
    weight_int8pack_mm nodes, for 8-bit weight-only quantization.

    Replaces dq(weight) -> linear(activation, dq)       with weight_int8pack_mm
    Replaces dq(weight) -> linear(activation, dq, bias) with weight_int8pack_mm -> add
    """

    def fuse_dequant_with_linear(
        self,
        graph_module: torch.fx.GraphModule,
        dequant_node: torch.fx.Node,
        linear_node: torch.fx.Node,
    ) -> None:
        activations = linear_node.args[0]
        bias = None
        if len(linear_node.args) > 2:
            bias = linear_node.args[2]
        quant_weight = dequant_node.args[0]
        scale = dequant_node.args[1]

        with graph_module.graph.inserting_before(linear_node):
            weight_int8pack_mm_node = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten._weight_int8pack_mm.default,
                (activations, quant_weight, scale),
            )
            if bias:
                add_node = graph_module.graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.add.Tensor,
                    (weight_int8pack_mm_node, bias),
                )
                linear_node.replace_all_uses_with(add_node)
            else:
                linear_node.replace_all_uses_with(weight_int8pack_mm_node)
            graph_module.graph.erase_node(linear_node)
            graph_module.graph.erase_node(dequant_node)

    def is_node_target(
        self, node: torch.fx.Node, target: torch._ops.OperatorBase
    ) -> bool:
        return node.op == "call_function" and node.target == target

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if self.is_node_target(node, exir_ops.edge.aten.linear.default):
                weight_node = node.args[1]
                if self.is_node_target(
                    weight_node,
                    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
                ):
                    # only fuse if weight tensor is int8 packed
                    quant_weight = weight_node.args[0]
                    if quant_weight.meta["val"].dtype != torch.int8:
                        continue
                    self.fuse_dequant_with_linear(graph_module, weight_node, node)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
