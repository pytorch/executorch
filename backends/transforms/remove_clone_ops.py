# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.remove_noop_pass import _DEQUANT_OPS, eliminate_dq_q


class RemoveCloneOpsTransform(ExportPass):
    """
    Trim the 'identity' operators to reduce the unnecessary copy overhead.
    """

    clone_ops: Set[torch._ops.OpOverload] = {
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    }

    def __init__(
        self,
        preserve_input_output_copies: bool = False,
        eliminate_quant_dequant_pairs: bool = True,
    ) -> None:
        super().__init__()
        self._preserve_input_output_copies = preserve_input_output_copies
        self._eliminate_quant_dequant_pairs = eliminate_quant_dequant_pairs

    def _remove(self, graph_module: torch.fx.GraphModule) -> bool:
        dequant_nodes = []
        modified = False

        for n in graph_module.graph.nodes:
            if n.target not in self.clone_ops:
                continue

            if self._is_non_identity_clone(n):
                continue

            # If preserve_input_output_copies is set, don't remove clones that directly
            # copy from input to output.
            if self._is_input_output_copy(n) and self._preserve_input_output_copies:
                continue

            modified = True
            to_be_removed = n
            for user_n in list(n.users.keys()):
                user_n.replace_input_with(n, n.args[0])
            if n.args[0].target in _DEQUANT_OPS:
                dequant_nodes += [n.args[0]]
            graph_module.graph.erase_node(to_be_removed)

        if self._eliminate_quant_dequant_pairs:
            eliminate_dq_q(graph_module, dequant_nodes)

        return modified

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        if self._remove(graph_module):
            graph_module.recompile()
            dead_code_elimination_pass(graph_module)
            return PassResult(graph_module, True)
        else:
            return PassResult(graph_module, False)

    def _is_non_identity_clone(self, node: torch.fx.Node) -> bool:
        """Return True if clone has modified memory layout or dim order."""

        # aten.clone: check for memory_format changes
        if node.target == exir_ops.edge.aten.clone.default:
            memory_format = node.kwargs.get("memory_format")
            if memory_format in (None, torch.preserve_format):
                return False
            input_meta = node.args[0].meta
            return "val" in input_meta and not input_meta["val"].is_contiguous(
                memory_format=memory_format
            )

        # _clone_dim_order: check for dim_order changes
        if node.target == exir_ops.edge.dim_order_ops._clone_dim_order.default:
            input_meta = node.args[0].meta
            return (
                "val" in node.meta
                and "val" in input_meta
                and node.meta["val"].dim_order() != input_meta["val"].dim_order()
            )

        return False

    def _is_input_output_copy(self, node: torch.fx.Node) -> bool:
        """Return True if the node input is a graph input and output goes into an output node."""

        input_node = node.args[0]
        if input_node.op != "placeholder":
            return False

        for users in node.users:
            if users.op == "output":
                return True

        return False
