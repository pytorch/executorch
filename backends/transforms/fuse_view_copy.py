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


def merge_view_copy_chains(graph: torch.fx.Graph) -> tuple[torch.fx.Graph, bool]:
    """
    Find chains of view_copy nodes and merge them into one view_copy node.
    Only merges view_copy nodes that are not used by any other nodes.
    """
    ops = exir_ops.edge
    view_op = ops.aten.view_copy.default
    modified = False
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
        graph_module.graph, merge_modified = merge_view_copy_chains(graph_module.graph)
        graph_module.graph, noop_modified = remove_noop_view_copy(graph_module.graph)
        modified = merge_modified or noop_modified
        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
