# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import torch
from executorch.exir.dialects._ops import ops
from torch.fx.passes.infra.pass_base import PassBase, PassResult


def _is_view_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.view_copy.default,
        ops.edge.aten.view_copy.default,
    )


def _maybe_remove_view_copy(node: torch.fx.Node) -> bool:
    assert _is_view_copy(node)

    # Remove node if all users are views
    for user in node.users:
        if not _is_view_copy(user):
            return False

    base = node.args[0]
    node.replace_all_uses_with(base)
    node.graph.erase_node(node)
    return True


class RemoveRedundantViewCopyPass(PassBase):
    """
    Removes redundant view_copy nodes.

    A view_copy is redundant if all of its users are view_copy.  Consider the
    following example:
        op1 -> view_copy1 -> view_copy2 -> view_copy3 -> op2.

    Provided view_copy1 and view_copy2 have no users outside the illustration
    above, we can remove them and shorten the graph to
        op1 -> view_copy3 -> op2.

    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        n_removed = 0  # number of redundant view_copy nodes removed
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue

            for node in module.graph.nodes:
                if _is_view_copy(node):
                    removed = _maybe_remove_view_copy(node)
                    if removed:
                        n_removed += 1
            module.recompile()

        logging.info(f"Removed {n_removed} view_copy nodes.")
        any_removed = n_removed > 0
        return PassResult(graph_module, any_removed)
