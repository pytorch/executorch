# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set, Type

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


UNARY_ELEMENTWISE_OPS = [
    exir_ops.edge.aten.view_copy.default,
    exir_ops.edge.aten.alias_copy.default,
    exir_ops.edge.aten.clone.default,
    exir_ops.edge.dim_order_ops._clone_dim_order.default,
    exir_ops.edge.aten._to_copy.default,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.ceil.default,
    exir_ops.edge.aten.floor.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.round.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.silu.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.tanh.default,
    exir_ops.edge.aten.sign.default,
    exir_ops.edge.aten.reciprocal.default,
    exir_ops.edge.aten.gelu.default,
    exir_ops.edge.aten.rsqrt.default,
    exir_ops.edge.aten.exp.default,
    exir_ops.edge.aten.log.default,
]


def merge_view_copy_chains(graph: torch.fx.Graph) -> tuple[torch.fx.Graph, bool]:
    """
    Find chains of view_copy nodes and unary elementwise ops and set all
    view_copy nodes to have the final shape. The views will then be removed
    by the remove_noop_view_copy call.

    Only merges view_copy nodes that are not used by any other nodes.
    """
    ops = exir_ops.edge
    view_op = ops.aten.view_copy.default
    modified = False
    for node in graph.nodes:
        if node.op == "call_function" and node.target == view_op:
            # Find a chain of unary elementwise ops and save all view_copy nodes
            end_node = node
            view_ops = [node]
            while (
                end_node.op == "call_function"
                and end_node.target in UNARY_ELEMENTWISE_OPS
                and len(end_node.users) == 1
                and list(end_node.users)[0].target in UNARY_ELEMENTWISE_OPS
            ):
                end_node = list(end_node.users)[0]
                if end_node.target == view_op:
                    view_ops.append(end_node)

            # Set all view_copy nodes to have the final shape
            if len(view_ops) > 1:
                final_shape = view_ops[-1].args[1]
                for node in view_ops:
                    new_args = (node.args[0], final_shape)
                    node.args = new_args
                modified = True

    graph.eliminate_dead_code()
    return graph, modified


def remove_noop_view_copy(graph: torch.fx.Graph) -> tuple[torch.fx.Graph, bool]:
    """
    Remove view_copy nodes that are no-ops.
    """
    ops = exir_ops.edge
    view_op = ops.aten.view_copy.default
    modified = False
    for node in graph.nodes:
        if node.op == "call_function" and node.target == view_op:
            input_shape = list(node.args[0].meta["val"].shape)
            target_shape = node.args[1]
            if input_shape == target_shape:
                node.replace_all_uses_with(node.args[0])
                modified = True
    graph.eliminate_dead_code()
    return graph, modified


class FuseViewCopyTransform(ExportPass):
    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module.graph, modified = merge_view_copy_chains(graph_module.graph)
        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        graph_module.graph, modified = remove_noop_view_copy(graph_module.graph)
        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
