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
    clone_op = exir_ops.edge.aten.clone.default
    clone_dim_order_op = exir_ops.edge.dim_order_ops._clone_dim_order.default

    for node in graph.nodes:
        if node.op != "call_function":
            continue

        # Identify clone_dim_order ops with unchanged memory layout.
        unchanged_layout_clone = (
            node.target == clone_dim_order_op
            and "val" in node.meta
            and "val" in node.args[0].meta
            and node.meta["val"].dim_order() == node.args[0].meta["val"].dim_order()
        )

        if node.target == clone_op or unchanged_layout_clone:
            with graph.inserting_after(node):
                node.replace_all_uses_with(node.args[0])

    graph.eliminate_dead_code()
    return graph


class RemoveCloneOpsTransform(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module.graph = remove_clone_ops(graph_module.graph)
        return PassResult(graph_module, True)
