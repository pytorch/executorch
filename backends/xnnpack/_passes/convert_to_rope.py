# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import logging

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.partition.graphs import rope
from executorch.exir.dialects._ops import ops as exir_ops

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

logger = logging.getLogger(__name__)


class _Layout(enum.Enum):
    BSHD = enum.auto()
    BHSD = enum.auto()


class ConvertToRopePass(XNNPACKPass):
    _BHSD_TO_BSHD_PERM = [0, 2, 1, 3]

    def _build_weights(
        self,
        graph_module: torch.fx.GraphModule,
        cos_node: torch.fx.Node,
        sin_node: torch.fx.Node,
        output_node: torch.fx.Node,
    ) -> torch.fx.Node:
        """
        Construct the XNNPACK RoPE weights tensor from cos and sin inputs.

        The most common HF RoPE pattern doubles the frequencies:
            cos/sin shape: [batch, seq, head_dim] where head_dim = 2 * (dim // 2)
            The first half and second half are identical.

        XNNPACK expects weights: [tokens, channels] where:
            weights[:, :C/2] = cos values (unique half)
            weights[:, C/2:] = sin values (unique half)

        We insert graph nodes to slice the unique halves and concatenate them.

        Note that this assumes that cos and sin come from a cat([x, x]) node for
        this to be sound. We check this in the pass.
        """
        head_dim = cos_node.meta["val"].shape[-1]
        half_dim = head_dim // 2

        with graph_module.graph.inserting_before(output_node):
            cos_half = graph_module.graph.call_function(
                exir_ops.edge.aten.slice_copy.Tensor,
                args=(cos_node, -1, 0, half_dim),
            )
            sin_half = graph_module.graph.call_function(
                exir_ops.edge.aten.slice_copy.Tensor,
                args=(sin_node, -1, 0, half_dim),
            )
            weights = graph_module.graph.call_function(
                exir_ops.edge.aten.cat.default,
                args=([cos_half, sin_half], -1),
            )

        return weights

    @staticmethod
    def _trace_through_unsqueezes(node: torch.fx.Node) -> torch.fx.Node:
        """Walk backwards through consecutive unsqueeze_copy ops to find the source."""
        current = node
        while (
            current.op == "call_function"
            and current.target == exir_ops.edge.aten.unsqueeze_copy.default
        ):
            current = current.args[0]
        return current

    @staticmethod
    def _find_trig_source(node: torch.fx.Node) -> torch.fx.Node | None:
        """Walk backwards through unsqueeze_copy ops to find cos/sin op."""
        current = node
        for _ in range(10):
            if current.op != "call_function":
                return None
            if current.target in (
                exir_ops.edge.aten.cos.default,
                exir_ops.edge.aten.sin.default,
            ):
                return current
            if current.target == exir_ops.edge.aten.unsqueeze_copy.default:
                current = current.args[0]
                continue
            return None
        return None

    @classmethod
    def _is_doubled_cat(cls, trig_node: torch.fx.Node) -> bool:
        """Check that a cos/sin node's input is cat(x, x) with identical args."""
        cat_node = trig_node.args[0]
        if (
            cat_node.op != "call_function"
            or cat_node.target != exir_ops.edge.aten.cat.default
        ):
            return False
        tensors = cat_node.args[0]
        return len(tensors) == 2 and tensors[0] is tensors[1]

    @classmethod
    def _has_doubled_freqs(
        cls,
        cos_unsqueezed: torch.fx.Node,
        sin_unsqueezed: torch.fx.Node,
    ) -> bool:
        """Verify that cos/sin frequencies are doubled (first half == second half).

        Traces back through unsqueeze_copy ops to find the cos/sin producer,
        then verifies its input is cat(x, x) where both args are the same
        node — a structural proof that the first and second halves are identical.
        """
        cos_trig = cls._find_trig_source(cos_unsqueezed)
        sin_trig = cls._find_trig_source(sin_unsqueezed)

        if cos_trig is None or sin_trig is None:
            return False

        return cls._is_doubled_cat(cos_trig) and cls._is_doubled_cat(sin_trig)

    @staticmethod
    def _trace_through_permute(node: torch.fx.Node) -> torch.fx.Node | None:
        """If node is a permute_copy that swaps dims 1 and 2, return its input."""
        if (
            node.op == "call_function"
            and node.target == exir_ops.edge.aten.permute_copy.default
            and list(node.args[1]) == [0, 2, 1, 3]
        ):
            return node.args[0]
        return None

    @staticmethod
    def _get_layout(cos_unsqueezed: torch.fx.Node) -> _Layout | None:
        """Determine the tensor layout from the cos unsqueeze dimension."""
        if not (
            cos_unsqueezed.op == "call_function"
            and cos_unsqueezed.target == exir_ops.edge.aten.unsqueeze_copy.default
        ):
            return None
        unsqueeze_dim = cos_unsqueezed.args[1]
        ndim = len(cos_unsqueezed.meta["val"].shape)
        normalized = unsqueeze_dim if unsqueeze_dim >= 0 else unsqueeze_dim + ndim
        if normalized == 2:
            return _Layout.BSHD
        if normalized == 1:
            return _Layout.BHSD
        return None

    def create_rope(
        self,
        graph_module: torch.fx.GraphModule,
        match: InternalMatch,
    ):
        logger.debug(f"Matched RoPE subgraph: {match}")

        # placeholder_nodes are in the order of the pattern's placeholder ops:
        # [x, cos_unsqueezed, sin_unsqueezed]
        x_node = match.placeholder_nodes[0]
        cos_unsqueezed = match.placeholder_nodes[1]
        sin_unsqueezed = match.placeholder_nodes[2]
        output_node = match.returning_nodes[0]

        # xnn_define_rope expects NTHC (batch, tokens, heads, channels) input.
        # BSHD (unsqueeze_dim=2) maps directly to NTHC.
        # BHSD (unsqueeze_dim=1) requires tracing through the BSHD→BHSD permute
        # to recover the BSHD input, then re-permuting the output back to BHSD.
        layout = self._get_layout(cos_unsqueezed)
        if layout == _Layout.BSHD:
            rope_input = x_node
        elif layout == _Layout.BHSD:
            rope_input = self._trace_through_permute(x_node)
            if rope_input is None:
                logger.debug("Skipping RoPE fusion: BHSD but x is not a permute_copy")
                return
        else:
            logger.debug("Skipping RoPE fusion: unrecognized layout")
            return

        cos_node = self._trace_through_unsqueezes(cos_unsqueezed)
        sin_node = self._trace_through_unsqueezes(sin_unsqueezed)

        if not self._has_doubled_freqs(cos_unsqueezed, sin_unsqueezed):
            logger.debug("Skipping RoPE fusion: cannot verify doubled frequencies")
            return

        weights = self._build_weights(graph_module, cos_node, sin_node, output_node)

        with graph_module.graph.inserting_before(output_node):
            rope_node = graph_module.graph.create_node(
                "call_function",
                torch.ops.xnnpack.rope.default,
                args=(rope_input, weights),
            )

            if layout == _Layout.BHSD:
                permute_node = graph_module.graph.call_function(
                    exir_ops.edge.aten.permute_copy.default,
                    args=(rope_node, self._BHSD_TO_BSHD_PERM),
                )
                result_node = permute_node
            else:
                result_node = rope_node

        output_node.replace_all_uses_with(result_node)
        graph_module.graph.eliminate_dead_code()

    # override
    def call(self, graph_module: torch.fx.GraphModule):
        total_matches = 0
        total_fused = 0
        for pattern in rope.get_graphs():
            sm = SubgraphMatcher(pattern.graph, ignore_literals=True)
            matches = list(sm.match(graph_module.graph))
            total_matches += len(matches)
            for match in matches:
                try:
                    self.create_rope(graph_module, match)
                    total_fused += 1
                except Exception:
                    logger.warning("Failed to fuse RoPE pattern", exc_info=True)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
