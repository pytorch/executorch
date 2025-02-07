# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def merge_view_copy_chains(graph: torch.fx.Graph) -> torch.fx.Graph:
    """
    Find chains of view_copy nodes and merge them into one view_copy node.
    Only merges view_copy nodes that are not used by any other nodes.
    """
    ops = exir_ops.edge
    view_op = ops.aten.view_copy.default
    for node in graph.nodes:
        if node.op == "call_function" and node.target == view_op:
            # find ending view_copy node in chain
            end_node = node
            while (
                end_node.op == "call_function"
                and end_node.target == view_op
                and len(end_node.users) == 1
                and list(end_node.users)[0].target == view_op
            ):
                end_node = list(end_node.users)[0]
            # we can swap the first node's shape arg with the last node's shape arg
            if node != end_node:
                with graph.inserting_after(node):
                    new_args = (node.args[0], end_node.args[1])
                    node.args = new_args
                    end_node.replace_all_uses_with(node)

    graph.eliminate_dead_code()
    return graph


class FuseViewCopyTransform(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module.graph = merge_view_copy_chains(graph_module.graph)
        return PassResult(graph_module, True)
