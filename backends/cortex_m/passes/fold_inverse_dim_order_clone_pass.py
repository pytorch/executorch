# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.fx
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx.passes.infra.pass_manager import PassResult


class FoldInverseDimOrderClonePass(ExportPass):
    """Fold adjacent `_clone_dim_order` pairs whose net effect is identity.

    The conv1d lowering inserts a `_clone_dim_order(dim_order=[0,2,3,1])`
    before each conv and a `_clone_dim_order(dim_order=[0,1,2,3])` after. When
    `FuseViewCopyTransform` collapses the intermediate view_copy chain between
    two consecutive conv1d lowerings, the surviving graph is

        ... -> _clone_dim_order(to NCHW) -> _clone_dim_order(to NHWC) -> ...

    where the second clone's dim_order is the inverse of the first applied to
    the same shape -- two byte reorders that cancel. This pass detects that
    exact pattern and replaces uses of the second clone with the first
    clone's input, then lets dead code elimination remove both.
    """

    _CLONE = exir_ops.edge.dim_order_ops._clone_dim_order.default

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target != self._CLONE:
                continue

            second_clone = node
            first_clone = second_clone.args[0]
            if (
                not isinstance(first_clone, torch.fx.Node)
                or first_clone.op != "call_function"
                or first_clone.target != self._CLONE
                or len(first_clone.users) != 1
            ):
                continue

            original_input = first_clone.args[0]
            if not isinstance(original_input, torch.fx.Node):
                continue

            # Net effect is identity iff the second clone's target dim_order
            # equals the input tensor's dim_order before the first clone.
            original_dim_order = tuple(original_input.meta["val"].dim_order())
            second_dim_order = tuple(second_clone.kwargs.get("dim_order", ()))
            if original_dim_order != second_dim_order:
                continue

            second_clone.replace_all_uses_with(original_input)
            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
