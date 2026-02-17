# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for SymInt preservation in ExportPass.

This test validates that _local_scalar_dense operations preserve their original
SymInt values during ExportPass retrace, which is critical for maintaining
constraint information (deferred_runtime_asserts) on unbacked symbols.

The issue: During retrace, slice operations create new FakeTensors without
item_memo, causing subsequent _local_scalar_dense calls to create new unbacked
symbols that lack the constraints of the original symbols. This leads to
GuardOnDataDependentSymNode errors when constraints are checked outside of
export mode.

The fix: ExportPass.call_operator preserves the original SymInt from meta["val"]
for _local_scalar_dense operations instead of re-executing the operation.
"""

import unittest
from typing import Tuple

import torch
from executorch.exir import to_edge
from executorch.exir.pass_base import ExportPass
from torch.export import Dim, export


class IdentityExportPass(ExportPass):
    """A minimal ExportPass that triggers retrace but makes no changes."""

    pass


class RoPEModule(torch.nn.Module):
    """
    Simplified RoPE (Rotary Position Embedding) module that demonstrates
    the _local_scalar_dense issue.

    The key pattern is:
    1. freqs_cos/freqs_sin are precomputed buffers of shape [max_seq_len, head_dim]
    2. At runtime, we slice them to [seq_len, head_dim] where seq_len is dynamic
    3. The slice uses start_pos (from input) which creates an unbacked symbol
    4. Constraint: start_pos + seq_len <= max_seq_len
    """

    def __init__(self, max_seq_len: int = 2048, head_dim: int = 32):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        # Precomputed RoPE frequencies
        self.register_buffer(
            "freqs_cos", torch.randn(max_seq_len, head_dim), persistent=False
        )
        self.register_buffer(
            "freqs_sin", torch.randn(max_seq_len, head_dim), persistent=False
        )

    def forward(
        self, x: torch.Tensor, start_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, head_dim]
            start_pos: Scalar tensor indicating the start position in the cache

        Returns:
            Tuple of (freqs_cos, freqs_sin) sliced to [seq_len, head_dim]
        """
        seq_len = x.shape[1]
        # This .item() call creates an unbacked symbol via _local_scalar_dense
        # The constraint start_pos + seq_len <= max_seq_len must be preserved
        pos = start_pos.item()
        # These constraints are critical for establishing bounds on the unbacked symbol:
        # - _check_is_size: ensures pos >= 0 (size-like semantics)
        # - _check: ensures pos + seq_len <= max_seq_len (slice bounds)
        torch._check_is_size(pos)
        torch._check(pos + seq_len <= self.max_seq_len)
        freqs_cos = self.freqs_cos[pos : pos + seq_len]
        freqs_sin = self.freqs_sin[pos : pos + seq_len]
        return freqs_cos, freqs_sin


class TestSymIntPreservation(unittest.TestCase):
    def test_rope_export_pass_symint_preservation(self):
        """
        Test that ExportPass preserves SymInt constraints for _local_scalar_dense.

        Before the fix, this would fail with:
        GuardOnDataDependentSymNode: Could not guard on data-dependent expression
        u1 + s1 > 2048

        After the fix, the original constrained symbol is preserved through retrace.
        """
        model = RoPEModule(max_seq_len=2048, head_dim=32)

        # Dynamic dimensions for export
        # Only seq_len is truly dynamic; batch is fixed at 1 since the model
        # doesn't use it in any operations that would make it dynamic
        seq_len = Dim("seq_len", min=1, max=2048)

        # Example inputs with explicit shapes
        x = torch.randn(1, 10, 32)  # [batch, seq_len, head_dim]
        start_pos = torch.tensor(0)

        # Export with dynamic shapes
        ep = export(
            model,
            (x, start_pos),
            dynamic_shapes={
                "x": {1: seq_len},  # Only seq_len is dynamic
                "start_pos": None,
            },
        )

        # Apply an identity ExportPass - this triggers retrace
        # Before the fix, this would fail with GuardOnDataDependentSymNode
        export_pass = IdentityExportPass()
        result = export_pass.call(ep.graph_module)

        # Verify the pass succeeded and graph is valid
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.graph_module)

        # Verify the graph contains item (the data-dependent operator)
        has_item_op = any(
            node.target in (torch.ops.aten.item.default, torch.ops.aten._local_scalar_dense.default)
            for node in result.graph_module.graph.nodes
            if node.op == "call_function"
        )
        self.assertTrue(
            has_item_op,
            "Graph should contain item or _local_scalar_dense operation",
        )

    def test_rope_to_edge_pipeline(self):
        """
        Test the full to_edge pipeline with RoPE-like patterns.

        This tests that the fix works in the full ExecuTorch lowering pipeline.
        """
        model = RoPEModule(max_seq_len=2048, head_dim=32)

        seq_len = Dim("seq_len", min=1, max=2048)

        x = torch.randn(1, 10, 32)
        start_pos = torch.tensor(0)

        ep = export(
            model,
            (x, start_pos),
            dynamic_shapes={
                "x": {1: seq_len},  # Only seq_len is dynamic
                "start_pos": None,
            },
        )

        # This should succeed without GuardOnDataDependentSymNode error
        edge_program = to_edge(ep)
        self.assertIsNotNone(edge_program)


if __name__ == "__main__":
    unittest.main()
