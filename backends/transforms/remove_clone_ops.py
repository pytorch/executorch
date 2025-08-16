# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def remove_clone_ops(graph: torch.fx.Graph) -> torch.fx.Graph:
    """
    Remove clone op nodes that have the same dim_order as their input, and replace their uses with the input node.
    """
    for node in graph.nodes:
        if node.op != "call_function":
            continue

        if is_unchanged_clone(node) or is_unchanged_dim_order_clone(node):
            with graph.inserting_after(node):
                node.replace_all_uses_with(node.args[0])

    graph.eliminate_dead_code()
    return graph


def is_unchanged_clone(node: torch.fx.Node) -> bool:
    """Determine if aten.clone has unchanged memory format."""
    if node.target != exir_ops.edge.aten.clone.default:
        return False

    memory_format = node.kwargs.get("memory_format")
    if memory_format in (None, torch.preserve_format):
        return True

    input_meta = node.args[0].meta
    return "val" in input_meta and input_meta["val"].is_contiguous(
        memory_format=memory_format
    )


def is_unchanged_dim_order_clone(node: torch.fx.Node) -> bool:
    """Determine if _clone_dim_order has unchanged dim order."""
    if node.target != exir_ops.edge.dim_order_ops._clone_dim_order.default:
        return False

    input_meta = node.args[0].meta
    return (
        "val" in node.meta
        and "val" in input_meta
        and node.meta["val"].dim_order() == input_meta["val"].dim_order()
    )


class RemoveCloneOpsTransform(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module.graph = remove_clone_ops(graph_module.graph)
        return PassResult(graph_module, True)
