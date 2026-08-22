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


class NormalizeViewCopyBasePass(PassBase):
    """
    Point each view_copy to the first upstream non-view.

    After this pass, the base of each view_copy is not a view_copy.

    When combined with dead-code elimination, this pass removes redundant
    view_copy nodes.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        n_updated = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if _is_view_copy(node):
                    base, size = node.args
                    if _is_view_copy(base):
                        # Point base to bases's base and update node's args
                        # Base's base will not be a view_copy because we iterate
                        # through the graph in topological order, replacing as we go.
                        base = base.args[0]
                        node.args = (base, size)
                        n_updated += 1

            module.recompile()

        logging.debug(f"Updated the base on {n_updated} view_copy nodes.")
        return PassResult(graph_module, n_updated > 0)

    def ensures(self, graph_module: torch.fx.GraphModule) -> None:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if _is_view_copy(node):
                    base, size = node.args
                    assert not _is_view_copy(base)
