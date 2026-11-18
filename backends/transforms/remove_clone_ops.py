# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set

import torch
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from executorch.exir.passes.remove_noop_pass import _DEQUANT_OPS, eliminate_dq_q


class RemoveCloneOpsTransform(ExportPass):
    """
    Trim the 'identity' operators to reduce the unnecessary copy overhead.

    First, every aten.clone or aten._to_copy whose only effect is a
    non-contiguous memory_format is deleted, and the graph is retraced so each
    node's dim order reflects the deletion. These ops carry a layout only in
    programs lowered without dim order, where the emitter cannot represent a
    non-contiguous layout. Dim-order copies (_clone_dim_order,
    _to_dim_order_copy) describe a layout the runtime honours and are left to
    the second step. Then clones that no longer change the layout are removed.
    A contiguous clone of an input that is not contiguous, such as a
    channels-last placeholder, stays.

    A non-contiguous layout copy that produces a graph output is not deleted.
    If its input does not already have that layout, it is an error: deleting it
    would change the layout callers expect for that output.
    """

    clone_ops: Set[torch._ops.OpOverload] = {
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    }

    layout_copy_ops: Set[torch._ops.OpOverload] = {
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten._to_copy.default,
    }

    def __init__(
        self,
        preserve_input_output_copies: bool = False,
        eliminate_quant_dequant_pairs: bool = True,
    ) -> None:
        super().__init__()
        self._preserve_input_output_copies = preserve_input_output_copies
        self._eliminate_quant_dequant_pairs = eliminate_quant_dequant_pairs

    def _remove_non_contiguous_layout_copies(
        self, graph_module: torch.fx.GraphModule
    ) -> bool:
        modified = False
        for n in list(graph_module.graph.nodes):
            if n.target not in self.layout_copy_ops:
                continue
            if not self._is_non_contiguous_layout_copy(n):
                continue
            if self._produces_output(n):
                continue
            n.replace_all_uses_with(n.args[0])
            graph_module.graph.erase_node(n)
            modified = True
        return modified

    def _check_output_layout_copies(self, graph_module: torch.fx.GraphModule) -> None:
        for n in graph_module.graph.nodes:
            if n.target not in self.layout_copy_ops:
                continue
            if not self._is_non_contiguous_layout_copy(n):
                continue
            if not self._produces_output(n):
                continue
            if n.args[0].meta["val"].dim_order() != n.meta["val"].dim_order():
                raise RuntimeError(
                    f"{n.format_node()} produces a graph output in a non-contiguous "
                    "layout. Removing it would change the layout of that output."
                )

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
        modified = False
        if self._remove_non_contiguous_layout_copies(graph_module):
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
            modified = True
        self._check_output_layout_copies(graph_module)
        if self._remove(graph_module):
            graph_module.recompile()
            dead_code_elimination_pass(graph_module)
            modified = True
        return PassResult(graph_module, modified)

    def _is_non_contiguous_layout_copy(self, node: torch.fx.Node) -> bool:
        """Return True if the node only copies its input into a non-contiguous layout."""

        memory_format = get_arg(node, "memory_format")
        if memory_format in (None, torch.preserve_format, torch.contiguous_format):
            return False

        # A copy that also converts the dtype is not a pure layout copy.
        if node.target == exir_ops.edge.aten._to_copy.default:
            dtype = get_arg(node, "dtype")
            if dtype is not None and dtype != node.args[0].meta["val"].dtype:
                return False
        return True

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

    def _produces_output(self, node: torch.fx.Node) -> bool:
        return any(user.op == "output" for user in node.users)

    def _is_input_output_copy(self, node: torch.fx.Node) -> bool:
        """Return True if the node input is a graph input and output goes into an output node."""

        input_node = node.args[0]
        if input_node.op != "placeholder":
            return False

        for users in node.users:
            if users.op == "output":
                return True

        return False
