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
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.backends.mlx.passes import get_default_passes
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


def _lower(model, inputs, transform_passes=None):
    return to_edge_transform_and_lower(
        export(model, inputs, strict=False),
        transform_passes=transform_passes,
        partitioner=[MLXPartitioner()],
    ).to_executorch()


def _delegate_count(program) -> int:
    return sum(
        1
        for node in program.exported_program().graph_module.graph.nodes
        if node.op == "call_function" and "executorch_call_delegate" in str(node.target)
    )


def _run(model, inputs, transform_passes=None):
    """Lower, execute, and return the node counts, the delegate count and the error.

    The delegate count is returned so a test can tell "decomposed onto this backend"
    apart from "not lowered here at all", which a node count alone cannot show.
    """
    with torch.no_grad():
        ref = model(*inputs)
    program = _lower(model, inputs, transform_passes)
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

    def test_normalization_runs_before_preservation_check(self):
        observed_ranks = []

        class RankCheckingPartitioner(MLXPartitioner):
            def ops_to_not_decompose(self, ep):
                for node in ep.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target
                        == torch.ops.aten.scaled_dot_product_attention.default
                    ):
                        observed_ranks.append(
                            tuple(arg.meta["val"].dim() for arg in node.args[:3])
                        )
                return super().ops_to_not_decompose(ep)

        for shape in ((16, 64), (2, 16, 64), (1, 2, 16, 64)):
            with self.subTest(rank=len(shape)):
                observed_ranks.clear()
                inputs = tuple(torch.randn(shape) for _ in range(3))
                program = to_edge_transform_and_lower(
                    export(Sdpa(), inputs, strict=False),
                    partitioner=[RankCheckingPartitioner()],
                )
                self.assertTrue(observed_ranks)
                self.assertTrue(all(ranks == (4, 4, 4) for ranks in observed_ranks))
                self.assertEqual(_delegate_count(program), 1)

    def test_handler_requires_normalized_rank4_inputs(self):
        for shape in ((16, 64), (2, 16, 64), (1, 2, 16, 64)):
            with self.subTest(rank=len(shape)):
                ep = export(Sdpa(), tuple(torch.randn(shape) for _ in range(3)))
                sdpa = next(
                    node
                    for node in ep.graph.nodes
                    if node.target
                    == torch.ops.aten.scaled_dot_product_attention.default
                )
                builder = MLXProgramBuilder(ep)
                builder.check_support_only()
                info = builder.node_info[sdpa]
                self.assertEqual(info.supported, len(shape) == 4)
                if len(shape) < 4:
                    self.assertIn("rank-4", info.unsupported_reason)

    def test_handler_keyword_options(self):
        inputs = tuple(torch.randn(1, 2, 3, 8) for _ in range(3))
        for scale in (None, 0.0, 0.25):
            with self.subTest(scale=scale):
                ep = export(Sdpa(), inputs)
                sdpa = next(
                    node
                    for node in ep.graph.nodes
                    if node.target
                    == torch.ops.aten.scaled_dot_product_attention.default
                )
                sdpa.kwargs = {
                    "query": sdpa.args[0],
                    "key": sdpa.args[1],
                    "value": sdpa.args[2],
                    "dropout_p": 0.0,
                    "is_causal": False,
                    "scale": scale,
                    "enable_gqa": False,
                }
                sdpa.args = ()
                ep.graph_module.recompile()
                built = MLXProgramBuilder(ep).build()
                instructions = [
                    instr.op
                    for chain in built.instruction_chains
                    for instr in chain.instructions
                    if type(instr.op).__name__ == "SdpaNode"
                ]
                self.assertEqual(len(instructions), 1)
                self.assertEqual(
                    instructions[0].scale, 8**-0.5 if scale is None else scale
                )

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
    """Repeated KV heads are fused only when the caller enables the shared passes."""

    def test_rank4_head_repeat_fusion_is_opt_in(self):
        for use_default_passes in (False, True):
            with self.subTest(use_default_passes=use_default_passes):
                counts, _, err = _run(
                    GroupedSdpa(dim=1),
                    (
                        torch.randn(2, 4, 16, 64),
                        torch.randn(2, 2, 16, 64),
                        torch.randn(2, 2, 16, 64),
                    ),
                    transform_passes=(
                        get_default_passes() if use_default_passes else None
                    ),
                )
                self.assertEqual(counts.get("SdpaNode", 0), 1)
                self.assertEqual(
                    counts.get("RepeatNode", 0), 0 if use_default_passes else 2
                )
                self.assertLess(err, 1e-4)

    def test_rank3_sequence_repeat_is_kept(self):
        # At rank 3 dim 1 is the key sequence, so absorbing the repeat would drop
        # half the keys. Without a mask that still sums correctly, which is what
        # makes it easy to miss; with a causal mask it is wrong by whole units.
        counts, _, err = _run(
            GroupedSdpa(dim=1, is_causal=True),
            (torch.randn(2, 16, 64), torch.randn(2, 8, 64), torch.randn(2, 8, 64)),
            transform_passes=get_default_passes(),
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

    def test_rank3_unequal_lengths_use_causal_correction(self):
        counts, delegates, err = _run(
            Sdpa(is_causal=True),
            (torch.randn(2, 6, 64), torch.randn(2, 16, 64), torch.randn(2, 16, 64)),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("SliceNode", 0), 2)
        self.assertGreater(delegates, 0)
        self.assertLess(err, 1e-4)

    def test_rank2_unequal_lengths_use_causal_correction(self):
        counts, _, err = _run(
            Sdpa(is_causal=True),
            (torch.randn(6, 64), torch.randn(16, 64), torch.randn(16, 64)),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("SliceNode", 0), 2)
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
