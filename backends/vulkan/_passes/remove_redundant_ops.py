# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set, Union

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

OpType = Union[str, torch._ops.OpOverload, EdgeOpOverload]


class RemoveRedundantOpsTransform(ExportPass):
    """
    Trim certain operators to reduce unnecessary overhead.
    """

    redundant_ops: Set[OpType] = {
        torch.clone,
        torch.ops.aten.clone.default,
        exir_ops.edge.aten.clone.default,
        torch.ops.aten.alias.default,
        exir_ops.edge.aten.alias.default,
        exir_ops.edge.aten.lift_fresh_copy.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    }

    def __init__(self) -> None:
        super(RemoveRedundantOpsTransform, self).__init__()

    def _should_remove(self, node: torch.fx.Node) -> bool:
        if node.target in self.redundant_ops:
            return True

        # Only remove to_copy if dtype does not change. Otherwise, memory format changes
        # will be handled internally by the backend.
        if (
            node.target == exir_ops.edge.aten._to_copy.default
            or node.target == torch.ops.aten._to_copy.default
        ):
            src_dtype = node.meta["val"].dtype
            # pyre-ignore
            dst_dtype = node.args[0].meta["val"].dtype
            return src_dtype == dst_dtype

        return False

    def _remove(self, graph_module: torch.fx.GraphModule) -> None:
        for node in graph_module.graph.nodes:
            if not self._should_remove(node):
                continue

            with graph_module.graph.inserting_after(node):
                node.replace_all_uses_with(node.args[0])

        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self._remove(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
