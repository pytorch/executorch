#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for graph transformation passes in the MLX backend.
"""

import unittest

import executorch.exir as exir
import torch
import torch.nn as nn
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.backends.mlx.passes import (
    _is_pure_dtype_cast,
    CanonicalizePermutePass,
    CollapseDtypeConversionPass,
    CollapsePermutePass,
    CollapseViewCopyPass,
    FuseRMSNormPass,
    RemoveNoOpsPass,
)
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.partitioner import PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _PreserveOpsPartitioner(MLXPartitioner):
    """MLXPartitioner that preserves ops (via ops_to_not_decompose) but skips delegation.

    This gives tests a real edge-dialect graph with MLX-relevant ops like
    ``item`` preserved, without delegating nodes to the MLX backend.
    """

    def partition(self, edge_program):
        return PartitionResult(
            tagged_exported_program=edge_program,
            partition_tags={},
        )


def _to_edge_gm(module, example_inputs, dynamic_shapes=None):
    """Export module and lower to edge dialect, returning the GraphModule."""
    ep = export(module, example_inputs, dynamic_shapes=dynamic_shapes, strict=False)
    edge = exir.to_edge_transform_and_lower(
        ep,
        partitioner=[_PreserveOpsPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    return edge.exported_program().graph_module


def _count_ops(gm, target):
    return sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and n.target == target
    )


def _find_nodes(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _has_op(gm, target):
    return _count_ops(gm, target) > 0


class TestIsPureDtypeCast(unittest.TestCase):

    def test_pure_dtype_only(self):
        self.assertTrue(_is_pure_dtype_cast({"dtype": torch.float16}))

    def test_dtype_with_none_kwargs(self):
        self.assertTrue(
            _is_pure_dtype_cast(
                {
                    "dtype": torch.float16,
                    "device": None,
                    "layout": None,
                }
            )
        )

    def test_dtype_with_non_none_memory_format(self):
        self.assertFalse(
            _is_pure_dtype_cast(
                {
                    "dtype": torch.float16,
                    "memory_format": torch.contiguous_format,
                }
            )
        )

    def test_dtype_with_non_none_device(self):
        self.assertFalse(
            _is_pure_dtype_cast(
                {
                    "dtype": torch.float16,
                    "device": torch.device("cpu"),
                }
            )
        )

    def test_no_dtype_key(self):
        self.assertFalse(_is_pure_dtype_cast({"device": None}))

    def test_empty_kwargs(self):
        self.assertFalse(_is_pure_dtype_cast({}))


class TestCanonicalizePermutePass(unittest.TestCase):

    def test_transpose_becomes_permute(self):
        class M(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1)

        gm = _to_edge_gm(M(), (torch.randn(3, 4),))
        transpose_target = exir_ops.edge.aten.transpose_copy.int

        if not _has_op(gm, transpose_target):
            self.skipTest("Edge lowering did not produce transpose_copy")

        result = CanonicalizePermutePass()(gm)

        self.assertTrue(result.modified)
        self.assertFalse(_has_op(result.graph_module, transpose_target))
        self.assertTrue(
            _has_op(result.graph_module, exir_ops.edge.aten.permute_copy.default)
        )

        nodes = _find_nodes(
            result.graph_module, exir_ops.edge.aten.permute_copy.default
        )
        self.assertEqual(nodes[0].args[1], [1, 0])

    def test_negative_dims_normalized(self):
        class M(nn.Module):
            def forward(self, x):
                return x.transpose(-2, -1)

        gm = _to_edge_gm(M(), (torch.randn(2, 3, 4),))
        result = CanonicalizePermutePass()(gm)

        nodes = _find_nodes(
            result.graph_module, exir_ops.edge.aten.permute_copy.default
        )
        self.assertEqual(len(nodes), 1)
        # transpose(-2, -1) on 3D → [0, 2, 1]
        self.assertEqual(nodes[0].args[1], [0, 2, 1])

    def test_noop_when_no_transpose(self):
        class M(nn.Module):
            def forward(self, x):
                return x + 1

        gm = _to_edge_gm(M(), (torch.randn(3, 4),))
        result = CanonicalizePermutePass()(gm)
        self.assertFalse(result.modified)


class TestCollapseViewCopyPass(unittest.TestCase):

    def test_consecutive_view_copys_collapsed(self):
        """view_copy(view_copy(x, s1), s2) → view_copy(x, s2)."""

        class M(nn.Module):
            def forward(self, x):
                return x.view(2, 6).view(3, 4)

        gm = _to_edge_gm(M(), (torch.randn(12),))

        target = exir_ops.edge.aten.view_copy.default
        before = _count_ops(gm, target)
        self.assertGreaterEqual(before, 2)

        result = CollapseViewCopyPass()(gm)

        self.assertTrue(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 1)

    def test_identity_view_copy_chain_removed(self):
        """view_copy(view_copy(x, s1), original_shape) → removes both."""

        class M(nn.Module):
            def forward(self, x):
                return x.view(12).view(3, 4)

        gm = _to_edge_gm(M(), (torch.randn(3, 4),))

        result = CollapseViewCopyPass()(gm)

        self.assertTrue(result.modified)
        self.assertEqual(
            _count_ops(result.graph_module, exir_ops.edge.aten.view_copy.default), 0
        )

    def test_single_view_copy_unchanged(self):
        class M(nn.Module):
            def forward(self, x):
                return x.view(12)

        gm = _to_edge_gm(M(), (torch.randn(3, 4),))

        result = CollapseViewCopyPass()(gm)
        self.assertFalse(result.modified)

    def test_collapse_with_dynamic_batch(self):
        """Consecutive view_copys with a dynamic leading dim should collapse."""
        from torch.export import Dim

        class M(nn.Module):
            def forward(self, x):
                return x.view(-1, 3, 4).view(-1, 2, 6)

        batch = Dim("batch", min=1, max=128)
        gm = _to_edge_gm(
            M(),
            (torch.randn(4, 12),),
            dynamic_shapes={"x": {0: batch}},
        )

        target = exir_ops.edge.aten.view_copy.default
        before = _count_ops(gm, target)
        self.assertGreaterEqual(before, 2)

        result = CollapseViewCopyPass()(gm)

        self.assertTrue(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 1)

    def test_identity_chain_with_dynamic_batch(self):
        """view_copy(view_copy(x, s1), original_shape) with dynamic dim → both removed."""
        from torch.export import Dim

        class M(nn.Module):
            def forward(self, x):
                return x.view(-1, 3, 4).view(-1, 12)

        batch = Dim("batch", min=1, max=128)
        gm = _to_edge_gm(
            M(),
            (torch.randn(4, 12),),
            dynamic_shapes={"x": {0: batch}},
        )

        target = exir_ops.edge.aten.view_copy.default
        before = _count_ops(gm, target)
        self.assertGreaterEqual(before, 2)

        result = CollapseViewCopyPass()(gm)
        self.assertTrue(result.modified)
        # Meta-shape comparison resolves SymInt identity → both view_copys removed
        self.assertEqual(_count_ops(result.graph_module, target), 0)


class TestCollapsePermutePass(unittest.TestCase):

    def test_inverse_permutations_removed(self):
        """permute(permute(x, p), inverse(p)) → identity → removed."""

        class M(nn.Module):
            def forward(self, x):
                return x.permute(2, 0, 1).permute(1, 2, 0)

        gm = _to_edge_gm(M(), (torch.randn(2, 3, 4),))
        target = exir_ops.edge.aten.permute_copy.default
        self.assertEqual(_count_ops(gm, target), 2)

        result = CollapsePermutePass()(gm)

        self.assertTrue(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 0)

    def test_non_identity_composed(self):
        """Non-identity composition yields a single permute."""

        class M(nn.Module):
            def forward(self, x):
                return x.permute(1, 0, 2).permute(0, 2, 1)

        gm = _to_edge_gm(M(), (torch.randn(2, 3, 4),))
        target = exir_ops.edge.aten.permute_copy.default
        self.assertEqual(_count_ops(gm, target), 2)

        result = CollapsePermutePass()(gm)

        self.assertTrue(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 1)

        # composed[i] = p1[p2[i]]  where p1=[1,0,2], p2=[0,2,1]
        # → [1, 2, 0]
        nodes = _find_nodes(result.graph_module, target)
        self.assertEqual(nodes[0].args[1], [1, 2, 0])

    def test_single_permute_unchanged(self):
        class M(nn.Module):
            def forward(self, x):
                return x.permute(1, 0, 2)

        gm = _to_edge_gm(M(), (torch.randn(2, 3, 4),))
        result = CollapsePermutePass()(gm)
        self.assertFalse(result.modified)

    def test_multi_user_parent_not_collapsed(self):
        """Don't collapse when the parent permute has multiple users."""

        class M(nn.Module):
            def forward(self, x):
                y = x.permute(1, 0, 2)
                a = y.permute(1, 0, 2)
                b = y.sum()
                return a + b

        gm = _to_edge_gm(M(), (torch.randn(2, 3, 4),))
        result = CollapsePermutePass()(gm)
        # Parent permute has 2 users → should not be collapsed
        self.assertFalse(result.modified)


class TestCollapseDtypeConversionPass(unittest.TestCase):

    def test_consecutive_casts_collapsed(self):
        """_to_copy(f32→bf16→f16) → _to_copy(f32→f16)."""

        class M(nn.Module):
            def forward(self, x):
                return x.to(torch.bfloat16).to(torch.float16)

        gm = _to_edge_gm(M(), (torch.randn(4, 4),))
        target = exir_ops.edge.aten._to_copy.default
        before = _count_ops(gm, target)

        if before < 2:
            self.skipTest("Export optimized away double cast")

        result = CollapseDtypeConversionPass()(gm)

        self.assertTrue(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 1)

        # Remaining cast should be to float16
        nodes = _find_nodes(result.graph_module, target)
        self.assertEqual(nodes[0].kwargs.get("dtype"), torch.float16)

    def test_single_cast_unchanged(self):
        class M(nn.Module):
            def forward(self, x):
                return x.to(torch.float16)

        gm = _to_edge_gm(M(), (torch.randn(4, 4),))
        result = CollapseDtypeConversionPass()(gm)
        self.assertFalse(result.modified)


class TestRemoveNoOpsPass(unittest.TestCase):

    def test_remove_clone(self):
        class M(nn.Module):
            def forward(self, x):
                return x.clone()

        gm = _to_edge_gm(M(), (torch.randn(4, 4),))
        target = exir_ops.edge.aten.clone.default

        if not _has_op(gm, target):
            self.skipTest("Export did not produce a clone op")

        result = RemoveNoOpsPass()(gm)

        self.assertTrue(result.modified)
        self.assertFalse(_has_op(result.graph_module, target))

    def test_remove_identity_view_copy(self):
        """view_copy(x, same_shape) → removed."""

        class M(nn.Module):
            def forward(self, x):
                return x.view(3, 4)

        gm = _to_edge_gm(M(), (torch.randn(3, 4),))
        target = exir_ops.edge.aten.view_copy.default

        if not _has_op(gm, target):
            self.skipTest("Export optimized away identity view_copy")

        result = RemoveNoOpsPass()(gm)

        self.assertTrue(result.modified)
        self.assertFalse(_has_op(result.graph_module, target))

    def test_remove_identity_permute(self):
        """permute_copy(x, [0, 1, ..., n-1]) → removed."""

        class M(nn.Module):
            def forward(self, x):
                return x.permute(0, 1, 2)

        gm = _to_edge_gm(M(), (torch.randn(2, 3, 4),))
        target = exir_ops.edge.aten.permute_copy.default

        if not _has_op(gm, target):
            self.skipTest("Export optimized away identity permute")

        result = RemoveNoOpsPass()(gm)

        self.assertTrue(result.modified)
        self.assertFalse(_has_op(result.graph_module, target))

    def test_identity_dtype_cast_removed_after_collapse(self):
        """Chain: f32→f16→f32 collapses to f32→f32, then RemoveNoOps removes it."""

        class M(nn.Module):
            def forward(self, x):
                return x.to(torch.float16).to(torch.float32)

        gm = _to_edge_gm(M(), (torch.randn(4, 4),))
        target = exir_ops.edge.aten._to_copy.default

        if _count_ops(gm, target) < 2:
            self.skipTest("Export optimized away double cast")

        CollapseDtypeConversionPass()(gm)
        result = RemoveNoOpsPass()(gm)

        self.assertTrue(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 0)

    def test_to_copy_with_memory_format_not_removed(self):
        """_is_pure_dtype_cast rejects kwargs with non-None memory_format."""
        # Can't easily produce this through export, so test the guard directly
        self.assertFalse(
            _is_pure_dtype_cast(
                {
                    "dtype": torch.float32,
                    "memory_format": torch.contiguous_format,
                }
            )
        )

    def test_non_identity_view_copy_kept(self):
        """view_copy to a different shape should NOT be removed."""

        class M(nn.Module):
            def forward(self, x):
                return x.view(6, 2)

        gm = _to_edge_gm(M(), (torch.randn(3, 4),))

        result = RemoveNoOpsPass()(gm)
        self.assertFalse(result.modified)

    def test_noop_when_nothing_to_remove(self):
        class M(nn.Module):
            def forward(self, x):
                return x + 1

        gm = _to_edge_gm(M(), (torch.randn(3, 4),))
        result = RemoveNoOpsPass()(gm)
        self.assertFalse(result.modified)

    def test_identity_view_copy_with_dynamic_batch(self):
        """view_copy(x, same_shape) with a dynamic dim → removed via meta-shape comparison."""
        from torch.export import Dim

        class M(nn.Module):
            def forward(self, x):
                return x.view(-1, 4)

        batch = Dim("batch", min=1, max=128)
        gm = _to_edge_gm(
            M(),
            (torch.randn(4, 4),),
            dynamic_shapes={"x": {0: batch}},
        )

        target = exir_ops.edge.aten.view_copy.default
        if not _has_op(gm, target):
            self.skipTest("Export optimized away identity view_copy")

        result = RemoveNoOpsPass()(gm)
        self.assertTrue(result.modified)
        self.assertFalse(_has_op(result.graph_module, target))

    def test_non_identity_view_copy_with_dynamic_batch(self):
        """view_copy(x, different_shape) with dynamic dim should be kept."""
        from torch.export import Dim

        class M(nn.Module):
            def forward(self, x):
                return x.view(-1, 2, 2)

        batch = Dim("batch", min=1, max=128)
        gm = _to_edge_gm(
            M(),
            (torch.randn(4, 4),),
            dynamic_shapes={"x": {0: batch}},
        )

        target = exir_ops.edge.aten.view_copy.default
        if not _has_op(gm, target):
            self.skipTest("Export did not produce view_copy")

        result = RemoveNoOpsPass()(gm)
        # Shape changes, so view_copy should be kept
        self.assertFalse(result.modified)

    def test_full_slice_with_dynamic_batch(self):
        """slice_copy shape comparison with dynamic dim should not crash."""
        from torch.export import Dim

        class M(nn.Module):
            def forward(self, x):
                a = x[:, :4]
                b = x[:, 4:]
                return torch.cat([b, a], dim=1)

        batch = Dim("batch", min=1, max=128)
        gm = _to_edge_gm(
            M(),
            (torch.randn(4, 8),),
            dynamic_shapes={"x": {0: batch}},
        )

        target = exir_ops.edge.aten.slice_copy.Tensor
        self.assertTrue(_has_op(gm, target), "Expected slice_copy in the graph")

        # Must not crash with symbolic shapes (input_val.shape has SymInt)
        RemoveNoOpsPass()(gm)


class TestFuseRMSNormPass(unittest.TestCase):

    def test_rms_norm_fused(self):
        """Decomposed RMSNorm should be fused into a single aten.rms_norm op."""

        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, x):
                variance = x.pow(2).mean(-1, keepdim=True)
                x = x * torch.rsqrt(variance + self.eps)
                return self.weight * x

        model = RMSNorm(16)
        model.eval()
        gm = _to_edge_gm(model, (torch.randn(1, 4, 16),))

        result = FuseRMSNormPass()(gm)

        self.assertTrue(
            result.modified, "FuseRMSNormPass should fuse the RMSNorm pattern"
        )

        has_rms_norm = any(
            n.op == "call_function" and "rms_norm" in str(n.target)
            for n in result.graph_module.graph.nodes
        )
        self.assertTrue(has_rms_norm)

        # Intermediate ops (pow, rsqrt, mean) should be removed
        has_rsqrt = any(
            n.op == "call_function" and "rsqrt" in str(n.target)
            for n in result.graph_module.graph.nodes
        )
        self.assertFalse(has_rsqrt)

    def test_noop_on_non_rms_norm(self):
        class M(nn.Module):
            def forward(self, x):
                return x + 1

        ep = export(M(), (torch.randn(4, 4),), strict=False)
        result = FuseRMSNormPass()(ep.graph_module)
        self.assertFalse(result.modified)


class TestPassComposition(unittest.TestCase):

    def test_collapse_view_copy(self):
        class M(nn.Module):
            def forward(self, x):
                return x.view(2, 6).view(3, 4)

        gm = _to_edge_gm(M(), (torch.randn(12),))
        target = exir_ops.edge.aten.view_copy.default

        self.assertGreaterEqual(_count_ops(gm, target), 2)

        CollapseViewCopyPass()(gm)
        self.assertEqual(_count_ops(gm, target), 1)

    def test_canonicalize_then_collapse_permute_identity(self):
        """Double transpose = identity → both removed."""

        class M(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).transpose(0, 1)

        gm = _to_edge_gm(M(), (torch.randn(3, 4),))
        target = exir_ops.edge.aten.permute_copy.default

        CanonicalizePermutePass()(gm)
        self.assertEqual(_count_ops(gm, target), 2)

        CollapsePermutePass()(gm)
        self.assertEqual(_count_ops(gm, target), 0)

    def test_full_pipeline_does_not_crash(self):
        """Running the full default pass list should not crash."""
        from executorch.backends.mlx.passes import get_default_passes

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(x).to(torch.float16)

        gm = _to_edge_gm(M(), (torch.randn(1, 16),))

        from executorch.exir.pass_base import ExportedProgramPassBase

        for p in get_default_passes():
            # EP-aware passes (e.g. MLXReinplacePass) require a full
            # ExportedProgram (graph_signature), not a bare GraphModule; they are
            # exercised in the reinplace tests.
            if isinstance(p, ExportedProgramPassBase):
                continue
            p(gm)

        gm.graph.lint()

    def test_correctness_after_all_passes(self):
        """Output values should be preserved after running all passes."""
        from executorch.backends.mlx.passes import get_default_passes

        class M(nn.Module):
            def forward(self, x):
                y = x.reshape(12).reshape(3, 4)
                return y.transpose(0, 1)

        module = M()
        module.eval()
        x = torch.randn(3, 4)
        expected = module(x)

        gm = _to_edge_gm(module, (x,))

        from executorch.exir.pass_base import ExportedProgramPassBase

        for p in get_default_passes():
            # EP-aware passes (e.g. MLXReinplacePass) require a full
            # ExportedProgram, not a bare GraphModule; skip here.
            if isinstance(p, ExportedProgramPassBase):
                continue
            p(gm)

        actual = gm(x)
        # Edge graph modules may return a tuple
        if isinstance(actual, tuple):
            actual = actual[0]
        torch.testing.assert_close(actual, expected)


class TestReinplacePass(unittest.TestCase):
    """MLXReinplacePass rewrites functional elementwise chains into in-place edge
    ops; the in-place handlers alias out == in to enable MLX buffer donation."""

    def _lower_and_get_ep(self, module, example_inputs, dynamic_shapes=None):
        from executorch.backends.mlx.passes import get_default_passes

        module.eval()
        ep = export(module, example_inputs, dynamic_shapes=dynamic_shapes, strict=False)
        edge = exir.to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )
        edge = edge.transform(get_default_passes())
        return edge.exported_program()

    def test_unary_chain_is_reinplaced_and_aliased(self):
        """exp(log(exp(x))): the dead middle temp is reinplaced and aliased."""
        from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder

        class M(nn.Module):
            def forward(self, x):
                return torch.exp(torch.log(torch.exp(x)))

        eep = self._lower_and_get_ep(M(), (torch.randn(4, 4),))

        # The pass introduced an in-place edge op for the dead temp (log_).
        targets = [str(n.target) for n in eep.graph.nodes if n.op == "call_function"]
        self.assertTrue(any("aten.log_" in t for t in targets), targets)

        # In the built MLX program, at least one link must alias out == in.
        g = MLXProgramBuilder(eep).build()
        aliased = []
        for chain in g.instruction_chains:
            for instr in chain.instructions:
                op = instr.op
                if type(op).__name__ in ("ExpNode", "LogNode"):
                    aliased.append(op.x.idx == op.out.idx)
        self.assertTrue(any(aliased), f"expected an in-place (out==in) link: {aliased}")

    def test_full_size_binary_is_reinplaced_and_aliased(self):
        """A full-size, dtype-matching dead-temp binary op is reinplaced+aliased."""
        from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder

        class M(nn.Module):
            def forward(self, x, y):
                h = torch.exp(x)  # full-size dead temp
                s = h + y  # intermediate full-size add
                return torch.neg(s)

        eep = self._lower_and_get_ep(M(), (torch.randn(4, 8), torch.randn(4, 8)))
        targets = [str(n.target) for n in eep.graph.nodes if n.op == "call_function"]
        self.assertTrue(any("aten.add_" in t for t in targets), targets)

        g = MLXProgramBuilder(eep).build()
        add_ops = [
            instr.op
            for chain in g.instruction_chains
            for instr in chain.instructions
            if type(instr.op).__name__ == "AddNode"
        ]
        self.assertTrue(
            any(op.a.idx == op.out.idx for op in add_ops),
            f"expected an in-place AddNode (out==a): {[(o.a.idx, o.out.idx) for o in add_ops]}",
        )

    def test_dynamic_shapes_lower_and_alias(self):
        """Reinplace must work (and not raise) under dynamic shapes: a full-size
        chain reinplaces+aliases, building the MLX program cleanly."""
        from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder

        class M(nn.Module):
            def forward(self, x, y):
                h = torch.exp(x)  # [B, D] dynamic B, dead temp
                s = h + y  # full-size, same symbol B
                return torch.neg(s)

        x = torch.randn(3, 8)
        y = torch.randn(3, 8)
        dynamic_shapes = {
            "x": {0: torch.export.Dim("B")},
            "y": {0: torch.export.Dim("B")},
        }
        eep = self._lower_and_get_ep(M(), (x, y), dynamic_shapes=dynamic_shapes)

        targets = [str(n.target) for n in eep.graph.nodes if n.op == "call_function"]
        self.assertTrue(any("aten.add_" in t for t in targets), targets)

        # Build must succeed and produce an in-place AddNode.
        g = MLXProgramBuilder(eep).build()
        add_ops = [
            instr.op
            for chain in g.instruction_chains
            for instr in chain.instructions
            if type(instr.op).__name__ == "AddNode"
        ]
        self.assertTrue(any(op.a.idx == op.out.idx for op in add_ops))

    def test_extra_ops_build_with_inplace_handlers(self):
        """clamp / pow / activations: the in-place edge op is produced and the
        MLX builder lowers it via the in-place handler (build succeeds).

        Numerics are covered by the upstream reinplace tests; here we only
        verify the MLX-specific path — that an in-place handler exists for each
        and the program builds.
        """
        import torch.nn.functional as F

        from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder

        cases = {
            "clamp": lambda x: torch.neg(torch.clamp(torch.exp(x), -1.0, 1.0)),
            "pow": lambda x: torch.neg(torch.exp(x) ** 2),
            "gelu": lambda x: torch.neg(F.gelu(torch.exp(x))),
            "relu": lambda x: torch.neg(torch.relu(torch.exp(x))),
            "leaky_relu": lambda x: torch.neg(F.leaky_relu(torch.exp(x), 0.1)),
            "hardtanh": lambda x: torch.neg(F.hardtanh(torch.exp(x))),
        }
        for name, fn in cases.items():
            with self.subTest(op=name):

                class M(nn.Module):
                    def __init__(self, fn):
                        super().__init__()
                        self.fn = fn

                    def forward(self, x):
                        return self.fn(x)

                eep = self._lower_and_get_ep(M(fn), (torch.randn(4, 8),))
                targets = [
                    str(n.target) for n in eep.graph.nodes if n.op == "call_function"
                ]
                # An in-place edge op (other than the terminal neg) is present.
                self.assertTrue(
                    any(
                        "aten." in t
                        and "_." in t.split("aten.")[-1]
                        and "neg_" not in t
                        for t in targets
                    ),
                    f"{name}: expected an in-place op, got {targets}",
                )
                # The MLX builder must lower the in-place op (handler registered).
                MLXProgramBuilder(eep).build()


if __name__ == "__main__":
    unittest.main()
