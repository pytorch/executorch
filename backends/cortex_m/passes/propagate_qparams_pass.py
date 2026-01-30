# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PropagateQParamsPass - Propagate qparams to consumer nodes.

This pass runs after FoldAndAnnotateQParamsPass to ensure ops like addmm
can find weight qparams even when the weight goes through a transpose/permute.

The issue: FoldAndAnnotateQParamsPass folds DQ into the passthrough node
(e.g., permute), storing qparams as input_qparams. But:
1. output_qparams is empty (no Q node after permute)
2. addmm's input_qparams[2] expects to find the weight qparams

This pass:
1. For passthrough ops: copies input_qparams to output_qparams (they're equal)
2. Propagates output_qparams from passthrough ops to addmm's input_qparams[2]
"""

from typing import cast

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import Node
from torch.fx.passes.infra.pass_manager import PassResult


class PropagateQParamsPass(ExportPass):
    """
    Propagates qparams from passthrough ops to their consumers.

    Specifically handles the case where weight goes through transpose/permute
    before reaching addmm, ensuring addmm has weight qparams at index 2.
    """

    PASSTHROUGH_OPS = {
        exir_ops.edge.aten.t.default,
        exir_ops.edge.aten.transpose.int,
        exir_ops.edge.aten.permute.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.view.default,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.reshape.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten.contiguous.default,
    }

    @staticmethod
    def _has_qparams(node: Node, key: str) -> bool:
        """Check if node has non-empty qparams for the given key."""
        return key in node.meta and len(node.meta.get(key, {})) > 0

    def _propagate_passthrough_qparams(self, node: Node) -> bool:
        """
        Propagate qparams through a passthrough op.

        For passthrough ops, input and output qparams are the same.
        Returns True if any modification was made.
        """
        modified = False
        input_node = node.args[0]

        if not isinstance(input_node, Node):
            return False

        # Propagate output_qparams from input to this node
        if self._has_qparams(input_node, "output_qparams"):
            if not self._has_qparams(node, "output_qparams"):
                node.meta["output_qparams"] = input_node.meta["output_qparams"]
                modified = True

        # Copy input_qparams to output_qparams (they're the same for passthrough)
        if self._has_qparams(node, "input_qparams"):
            if not self._has_qparams(node, "output_qparams"):
                node.meta["output_qparams"] = {0: node.meta["input_qparams"][0]}
                modified = True

        # Copy output_qparams to input_qparams if missing
        if self._has_qparams(node, "output_qparams"):
            if "input_qparams" not in node.meta:
                node.meta["input_qparams"] = {}
            if 0 not in node.meta["input_qparams"]:
                node.meta["input_qparams"][0] = node.meta["output_qparams"][0]
                modified = True

        return modified

    def _propagate_addmm_weight_qparams(self, node: Node) -> bool:
        """
        Propagate weight qparams to addmm node at index 2.

        addmm(bias, input, weight.T) expects weight qparams at index 2.
        Returns True if any modification was made.
        """
        if len(node.args) < 3:
            return False

        if "input_qparams" not in node.meta:
            node.meta["input_qparams"] = {}

        if 2 in node.meta["input_qparams"]:
            return False

        weight_node = node.args[2]
        if not isinstance(weight_node, Node):
            return False

        if not self._has_qparams(weight_node, "output_qparams"):
            return False

        if 0 not in weight_node.meta["output_qparams"]:
            return False

        node.meta["input_qparams"][2] = weight_node.meta["output_qparams"][0]
        return True

    def call(self, graph_module):
        modified = False

        # First pass: Propagate qparams through passthrough ops
        for node in graph_module.graph.nodes:
            node = cast(Node, node)

            if node.op != "call_function":
                continue
            if node.target not in self.PASSTHROUGH_OPS:
                continue
            if len(node.args) == 0:
                continue

            if self._propagate_passthrough_qparams(node):
                modified = True

        # Second pass: Propagate qparams to addmm nodes
        for node in graph_module.graph.nodes:
            node = cast(Node, node)

            if node.op != "call_function":
                continue
            if node.target != exir_ops.edge.aten.addmm.default:
                continue

            if self._propagate_addmm_weight_qparams(node):
                modified = True

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)
