#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the MLX partitioner.
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.backends.mlx.test.test_utils import get_mlx_node_counts
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.runtime import Runtime
from torch.export import export


class TestMLXPartitionerRejectsToEdge(unittest.TestCase):
    """MLXPartitioner must only be used via to_edge_transform_and_lower."""

    def test_to_edge_then_to_backend_raises(self):
        class M(nn.Module):
            def forward(self, x):
                return x + 1

        ep = export(M(), (torch.randn(4),), strict=False)
        edge = to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )

        with self.assertRaises(RuntimeError) as ctx:
            edge.to_backend(MLXPartitioner())

        self.assertIn("to_edge_transform_and_lower", str(ctx.exception))


def _lower(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs, strict=False),
        partitioner=[MLXPartitioner()],
    ).to_executorch()


def _delegate_count(program) -> int:
    return sum(
        1
        for node in program.exported_program().graph_module.graph.nodes
        if node.op == "call_function" and "executorch_call_delegate" in str(node.target)
    )


def _run(model, inputs):
    """Lower, execute, and return the node counts, the delegate count and the error.

    The delegate count is returned so a test can tell "decomposed onto this backend"
    apart from "not lowered here at all", which a node count alone cannot show.
    """
    with torch.no_grad():
        ref = model(*inputs)
    program = _lower(model, inputs)
    delegates = _delegate_count(program)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pte"
        path.write_bytes(program.buffer)
        counts = get_mlx_node_counts(path)
        method = Runtime.get().load_program(path).load_method("forward")
        out = method.execute(list(inputs))[0]
    return counts, delegates, (out - ref).abs().max().item()


class Sdpa(nn.Module):
    def __init__(self, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )


class GroupedSdpa(nn.Module):
    """Grouped key/value attention, where the repeat is unwrapped before the kernel."""

    def __init__(self, dim: int, is_causal: bool = False):
        super().__init__()
        self.dim = dim
        self.is_causal = is_causal

    def forward(self, q, k, v):
        k = k.repeat_interleave(2, dim=self.dim)
        v = v.repeat_interleave(2, dim=self.dim)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )


class TestMLXPartitionerSdpaShapes(unittest.TestCase):
    """The fused kernel takes rank 4, so other ranks are adapted or left alone."""

    def test_rank4_is_unchanged(self):
        counts, _, err = _run(
            Sdpa(), tuple(torch.randn(2, 4, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 0)
        self.assertEqual(counts.get("SqueezeNode", 0), 0)
        self.assertLess(err, 1e-4)

    def test_rank3_is_lifted_once(self):
        counts, _, err = _run(Sdpa(), tuple(torch.randn(2, 16, 64) for _ in range(3)))
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 3)
        self.assertEqual(counts.get("SqueezeNode", 0), 1)
        self.assertLess(err, 1e-4)

    def test_rank2_is_lifted_twice(self):
        counts, _, err = _run(Sdpa(), tuple(torch.randn(16, 64) for _ in range(3)))
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 6)
        self.assertEqual(counts.get("SqueezeNode", 0), 1)
        self.assertLess(err, 1e-4)

    def test_rank5_is_decomposed_on_this_backend(self):
        # Folding the leading dimensions pairs the wrong operands once one of them
        # broadcasts a batch, so this decomposes rather than fusing.
        counts, delegates, err = _run(
            Sdpa(), tuple(torch.randn(2, 2, 4, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 0)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)

    def test_unequal_batch_is_decomposed_on_this_backend(self):
        counts, delegates, err = _run(
            Sdpa(),
            (
                torch.randn(2, 4, 16, 64),
                torch.randn(1, 4, 16, 64),
                torch.randn(1, 4, 16, 64),
            ),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 0)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)

    def test_zero_head_count_declines_instead_of_raising(self):
        # The head multiple test would divide by zero here, and raising from the
        # matcher aborts the whole export rather than declining this one node. Only
        # lowering is checked: a zero-size operand is not executable either way.
        program = _lower(
            Sdpa(),
            (
                torch.randn(1, 4, 8, 16),
                torch.randn(1, 0, 8, 16),
                torch.randn(1, 0, 8, 16),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pte"
            path.write_bytes(program.buffer)
            self.assertEqual(get_mlx_node_counts(path).get("SdpaNode", 0), 0)


class TestMLXPartitionerGroupedKeys(unittest.TestCase):
    """The grouped key/value unwrap reads dim 1 as the head, which holds at rank 4."""

    def test_rank4_head_repeat_is_absorbed(self):
        counts, _, err = _run(
            GroupedSdpa(dim=1),
            (
                torch.randn(2, 4, 16, 64),
                torch.randn(2, 2, 16, 64),
                torch.randn(2, 2, 16, 64),
            ),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("RepeatNode", 0), 0)
        self.assertLess(err, 1e-4)

    def test_rank3_sequence_repeat_is_kept(self):
        # At rank 3 dim 1 is the key sequence, so absorbing the repeat would drop
        # half the keys. Without a mask that still sums correctly, which is what
        # makes it easy to miss; with a causal mask it is wrong by whole units.
        counts, _, err = _run(
            GroupedSdpa(dim=1, is_causal=True),
            (torch.randn(2, 16, 64), torch.randn(2, 8, 64), torch.randn(2, 8, 64)),
        )
        self.assertEqual(counts.get("RepeatNode", 0), 2)
        self.assertLess(err, 1e-4)


class TestMLXPartitionerSdpaCausal(unittest.TestCase):
    """MLX anchors a causal mask at the bottom right and torch at the top left."""

    def test_equal_lengths_stay_fused(self):
        counts, _, err = _run(
            Sdpa(is_causal=True), tuple(torch.randn(1, 4, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertLess(err, 1e-4)

    def test_rank3_equal_lengths_are_lifted(self):
        counts, _, err = _run(
            Sdpa(is_causal=True), tuple(torch.randn(2, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 3)
        self.assertLess(err, 1e-4)

    def test_rank3_unequal_lengths_are_not_lifted(self):
        # The two conventions disagree here and the disagreement is silent, so a
        # shape this backend could not previously reach is not opened up.
        counts, delegates, err = _run(
            Sdpa(is_causal=True),
            (torch.randn(2, 6, 64), torch.randn(2, 16, 64), torch.randn(2, 16, 64)),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 0)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)

    def test_rank2_unequal_lengths_are_not_lifted(self):
        counts, _, err = _run(
            Sdpa(is_causal=True),
            (torch.randn(6, 64), torch.randn(16, 64), torch.randn(16, 64)),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 0)
        self.assertLess(err, 1e-4)


class TestMLXPartitionerMixedSupport(unittest.TestCase):
    """An operator is preserved from decomposition per operator, not per call."""

    def test_supported_and_unsupported_calls_in_one_graph(self):
        class Mixed(nn.Module):
            def forward(self, a, b):
                x = torch.nn.functional.scaled_dot_product_attention(a, a, a)
                y = torch.nn.functional.scaled_dot_product_attention(b, b, b)
                return x.sum() + y.sum()

        # Without giving the whole operator back, the rank-5 call would be neither
        # lowered nor decomposed and this would raise a missing out variant.
        counts, delegates, err = _run(
            Mixed().eval(),
            (torch.randn(1, 4, 16, 64), torch.randn(2, 2, 4, 16, 64)),
        )
        # The cost of the coarse choice: the supported call is unfused as well.
        self.assertEqual(counts.get("SdpaNode", 0), 0)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-3)

    def test_mixed_support_outside_attention(self):
        class TwoRolls(nn.Module):
            def forward(self, x):
                return torch.roll(x, 1, dims=0).sum() + torch.roll(x, 1).sum()

        _, delegates, err = _run(TwoRolls().eval(), (torch.randn(4, 8),))
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)


if __name__ == "__main__":
    unittest.main()
