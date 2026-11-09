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
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.backends.mlx.passes import (
    _is_pure_dtype_cast,
    CanonicalizePermutePass,
    CollapseDtypeConversionPass,
    CollapsePermutePass,
    CollapseViewCopyPass,
    FuseRMSNormPass,
    get_default_passes,
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


def _to_edge(module, example_inputs, dynamic_shapes=None):
    """Preserve MLX-supported ops without delegating, for pass inspection."""
    ep = export(module, example_inputs, dynamic_shapes=dynamic_shapes, strict=False)
    edge = exir.to_edge_transform_and_lower(
        ep,
        partitioner=[_PreserveOpsPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    return edge


def _to_edge_gm(module, example_inputs, dynamic_shapes=None):
    return (
        _to_edge(module, example_inputs, dynamic_shapes).exported_program().graph_module
    )


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

    def test_lossy_consecutive_casts_kept(self):
        class M(nn.Module):
            def __init__(self, intermediate, output):
                super().__init__()
                self.intermediate = intermediate
                self.output = output

            def forward(self, x):
                return x.to(self.intermediate).to(self.output)

        floats = torch.tensor([-1.003, -0.9, 0.9, 1.003], dtype=torch.float32)
        cases = (
            (floats, torch.int32, torch.float32),
            (floats, torch.bool, torch.float32),
            (floats, torch.bfloat16, torch.float16),
            (floats, torch.float16, torch.float32),
            (
                torch.tensor([2**24 + 1, 2**24 + 3, -(2**24 + 1)]),
                torch.float32,
                torch.int64,
            ),
        )
        target = exir_ops.edge.aten._to_copy.default
        for x, intermediate, output in cases:
            with self.subTest(source=x.dtype, intermediate=intermediate, output=output):
                model = M(intermediate, output)
                expected = model(x)
                self.assertFalse(torch.equal(expected, x.to(output)))
                gm = _to_edge_gm(model, (x,))
                self.assertEqual(_count_ops(gm, target), 2)

                result = CollapseDtypeConversionPass()(gm)

                self.assertFalse(result.modified)
                self.assertEqual(_count_ops(result.graph_module, target), 2)
                result.graph_module.recompile()
                torch.testing.assert_close(
                    result.graph_module(x)[0], expected, rtol=0, atol=0
                )

    def test_lossless_widening_casts_collapsed(self):
        class M(nn.Module):
            def __init__(self, intermediate, output):
                super().__init__()
                self.intermediate = intermediate
                self.output = output

            def forward(self, x):
                return x.to(self.intermediate).to(self.output)

        cases = (
            (torch.float16, torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float32, torch.float16),
            (torch.float16, torch.float64, torch.bfloat16),
            (torch.bfloat16, torch.float64, torch.float16),
            (torch.float32, torch.float64, torch.float16),
        )
        target = exir_ops.edge.aten._to_copy.default
        for source, intermediate, output in cases:
            with self.subTest(source=source, intermediate=intermediate):
                x = torch.tensor([-1.003, -0.9, 0.9, 1.003], dtype=source)
                model = M(intermediate, output)
                gm = _to_edge_gm(model, (x,))
                self.assertEqual(_count_ops(gm, target), 2)
                source_node = _find_nodes(gm, target)[0].args[0]

                result = CollapseDtypeConversionPass()(gm)

                self.assertTrue(result.modified)
                nodes = _find_nodes(result.graph_module, target)
                self.assertEqual(len(nodes), 1)
                self.assertIs(nodes[0].args[0], source_node)
                self.assertEqual(nodes[0].kwargs["dtype"], output)
                result.graph_module.recompile()
                torch.testing.assert_close(
                    result.graph_module(x)[0], model(x), rtol=0, atol=0
                )

    def test_boolean_to_float_casts_collapsed(self):
        class M(nn.Module):
            def __init__(self, intermediate, output):
                super().__init__()
                self.intermediate = intermediate
                self.output = output

            def forward(self, x):
                return x.to(self.intermediate).to(self.output)

        x = torch.tensor([False, True])
        target = exir_ops.edge.aten._to_copy.default
        for intermediate in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            for output in (torch.float16, torch.float32, torch.int32):
                if intermediate == output:
                    continue
                with self.subTest(intermediate=intermediate, output=output):
                    model = M(intermediate, output)
                    gm = _to_edge_gm(model, (x,))
                    nodes = _find_nodes(gm, target)
                    self.assertEqual(len(nodes), 2)
                    source_node = nodes[0].args[0]

                    result = CollapseDtypeConversionPass()(gm)

                    self.assertTrue(result.modified)
                    nodes = _find_nodes(result.graph_module, target)
                    self.assertEqual(len(nodes), 1)
                    self.assertIs(nodes[0].args[0], source_node)
                    self.assertEqual(nodes[0].kwargs["dtype"], output)
                    result.graph_module.recompile()
                    torch.testing.assert_close(
                        result.graph_module(x)[0], model(x), rtol=0, atol=0
                    )

    def test_missing_source_metadata_not_collapsed(self):
        class M(nn.Module):
            def forward(self, x):
                return x.to(torch.float32).to(torch.bfloat16)

        x = torch.tensor([1.003, -0.9], dtype=torch.float16)
        gm = _to_edge_gm(M(), (x,))
        target = exir_ops.edge.aten._to_copy.default
        nodes = _find_nodes(gm, target)
        self.assertEqual(len(nodes), 2)
        del nodes[0].args[0].meta["val"]

        result = CollapseDtypeConversionPass()(gm)

        self.assertFalse(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 2)
        torch.testing.assert_close(result.graph_module(x)[0], M()(x), rtol=0, atol=0)

    def test_multi_user_parent_not_collapsed(self):
        class M(nn.Module):
            def forward(self, x):
                y = x.to(torch.float32)
                return y, y.to(torch.bfloat16)

        x = torch.tensor([1.003, -0.9], dtype=torch.float16)
        gm = _to_edge_gm(M(), (x,))
        target = exir_ops.edge.aten._to_copy.default
        nodes = _find_nodes(gm, target)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(nodes[0].users), 2)

        result = CollapseDtypeConversionPass()(gm)

        self.assertFalse(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 2)
        torch.testing.assert_close(result.graph_module(x), M()(x), rtol=0, atol=0)

    def test_non_pure_cast_not_collapsed(self):
        class M(nn.Module):
            def forward(self, x):
                return x.to(torch.float32).to(torch.bfloat16)

        target = exir_ops.edge.aten._to_copy.default
        for cast_index in (0, 1):
            with self.subTest(cast_index=cast_index):
                gm = _to_edge_gm(M(), (torch.ones(2, dtype=torch.float16),))
                nodes = _find_nodes(gm, target)
                self.assertEqual(len(nodes), 2)
                node = nodes[cast_index]
                node.kwargs = dict(node.kwargs, memory_format=torch.contiguous_format)

                result = CollapseDtypeConversionPass()(gm)

                self.assertFalse(result.modified)
                self.assertEqual(_count_ops(result.graph_module, target), 2)

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

    def test_lossy_dtype_roundtrip_kept_after_collapse(self):
        class M(nn.Module):
            def forward(self, x):
                return x.to(torch.float16).to(torch.float32)

        x = torch.tensor([-1.003, -0.9, 0.9, 1.003], dtype=torch.float32)
        expected = M()(x)
        self.assertFalse(torch.equal(expected, x))
        gm = _to_edge_gm(M(), (x,))
        target = exir_ops.edge.aten._to_copy.default
        self.assertEqual(_count_ops(gm, target), 2)

        collapsed = CollapseDtypeConversionPass()(gm)
        result = RemoveNoOpsPass()(collapsed.graph_module)

        self.assertFalse(collapsed.modified)
        self.assertFalse(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 2)
        result.graph_module.recompile()
        torch.testing.assert_close(result.graph_module(x)[0], expected, rtol=0, atol=0)

    def test_lossless_dtype_roundtrip_removed_after_collapse(self):
        class M(nn.Module):
            def forward(self, x):
                return x.to(torch.float32).to(torch.float16)

        x = torch.tensor([-1.003, -0.9, 0.9, 1.003], dtype=torch.float16)
        gm = _to_edge_gm(M(), (x,))
        target = exir_ops.edge.aten._to_copy.default
        self.assertEqual(_count_ops(gm, target), 2)

        collapsed = CollapseDtypeConversionPass()(gm)
        self.assertTrue(collapsed.modified)
        self.assertEqual(_count_ops(collapsed.graph_module, target), 1)
        result = RemoveNoOpsPass()(collapsed.graph_module)

        self.assertTrue(result.modified)
        self.assertEqual(_count_ops(result.graph_module, target), 0)
        result.graph_module.recompile()
        torch.testing.assert_close(result.graph_module(x)[0], M()(x), rtol=0, atol=0)

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
        inputs = (torch.randn(1, 4, 16),)
        result = (
            _to_edge(model, inputs).transform([FuseRMSNormPass()]).exported_program()
        )
        torch.testing.assert_close(result.module()(*inputs), model(*inputs))

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

        inputs = (torch.randn(4, 4),)
        edge = _to_edge(M(), inputs)
        before = str(edge.exported_program().graph)
        result = edge.transform([FuseRMSNormPass()]).exported_program()
        self.assertEqual(str(result.graph), before)
        torch.testing.assert_close(result.module()(*inputs), M()(*inputs))


class TestPassComposition(unittest.TestCase):

    def test_collapse_view_copy(self):
        class M(nn.Module):
            def forward(self, x):
                return x.view(2, 6).view(3, 4)

        edge = _to_edge(M(), (torch.randn(12),))
        target = exir_ops.edge.aten.view_copy.default

        self.assertGreaterEqual(
            _count_ops(edge.exported_program().graph_module, target), 2
        )
        result = edge.transform([CollapseViewCopyPass()]).exported_program()
        self.assertEqual(_count_ops(result.graph_module, target), 1)

    def test_canonicalize_then_collapse_permute_identity(self):
        """Double transpose = identity → both removed."""

        class M(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).transpose(0, 1)

        edge = _to_edge(M(), (torch.randn(3, 4),))
        target = exir_ops.edge.aten.permute_copy.default

        edge = edge.transform([CanonicalizePermutePass()])
        self.assertEqual(_count_ops(edge.exported_program().graph_module, target), 2)
        result = edge.transform([CollapsePermutePass()]).exported_program()
        self.assertEqual(_count_ops(result.graph_module, target), 0)

    def test_full_pipeline_does_not_crash(self):
        """Run both GraphModule and ExportedProgram passes and build the result."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(x).to(torch.float16)

        model = M().eval()
        inputs = (torch.randn(1, 16),)
        result = (
            _to_edge(model, inputs).transform(get_default_passes()).exported_program()
        )
        result.graph.lint()
        torch.testing.assert_close(result.module()(*inputs), model(*inputs))
        MLXProgramBuilder(result).build()

    def test_boolean_cache_reset_uses_one_cast(self):
        class M(nn.Module):
            def forward(self, input_pos):
                reset = (input_pos[0] == 0).to(torch.bfloat16).to(torch.float32)
                return 1.0 - reset

        model = M()
        result = (
            _to_edge(model, (torch.tensor([0]),))
            .transform(get_default_passes())
            .exported_program()
        )
        result.graph.lint()
        for position in (0, 7):
            with self.subTest(position=position):
                input_pos = torch.tensor([position])
                torch.testing.assert_close(
                    result.module()(input_pos), model(input_pos), rtol=0, atol=0
                )
        built = MLXProgramBuilder(result).build()
        self.assertEqual(
            sum(
                type(instr.op).__name__ == "AsTypeNode"
                for chain in built.instruction_chains
                for instr in chain.instructions
            ),
            1,
        )

    def test_correctness_after_all_passes(self):
        """Output values should be preserved after running all passes."""

        class M(nn.Module):
            def forward(self, x):
                y = x.reshape(12).reshape(3, 4)
                return y.transpose(0, 1)

        module = M()
        module.eval()
        x = torch.randn(3, 4)
        expected = module(x)

        result = (
            _to_edge(module, (x,)).transform(get_default_passes()).exported_program()
        )
        torch.testing.assert_close(result.module()(x), expected)


class TestDefaultFusionPipeline(unittest.TestCase):
    """Check shared transforms compose with MLX cleanup and builder dispatch."""

    def _run_pipeline(self, model, inputs, expected_op, **tolerances):
        model.eval()
        with torch.no_grad():
            expected = model(*inputs)
        result = (
            _to_edge(model, inputs).transform(get_default_passes()).exported_program()
        )
        result.graph.lint()
        with torch.no_grad():
            actual = result.module()(*inputs)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, **tolerances)
        built = MLXProgramBuilder(result).build()
        self.assertEqual(
            sum(
                type(instr.op).__name__ == expected_op
                for chain in built.instruction_chains
                for instr in chain.instructions
            ),
            1,
        )
        return result

    def test_repeated_kv_gqa(self):
        class GQA(nn.Module):
            def forward(self, q, k, v, mask):
                def repeat(x):
                    shape = list(x.shape)
                    expanded = [*shape[:-2], 2, *shape[-2:]]
                    shape[-3] *= 2
                    return x.unsqueeze(-3).expand(expanded).reshape(shape)

                return nn.functional.scaled_dot_product_attention(
                    q, repeat(k), repeat(v), mask, scale=0.25
                )

        for rank in (3, 4):
            with self.subTest(rank=rank):
                prefix = (2,) if rank == 4 else ()
                inputs = (
                    torch.randn(*prefix, 4, 3, 8),
                    torch.randn(*prefix, 2, 5, 8),
                    torch.randn(*prefix, 2, 5, 8),
                    torch.randn(3, 5),
                )
                result = self._run_pipeline(GQA(), inputs, "SdpaNode")
                sdpa_nodes = _find_nodes(
                    result.graph_module,
                    exir_ops.edge.aten.scaled_dot_product_attention.default,
                )
                self.assertEqual(len(sdpa_nodes), 1)
                sdpa = sdpa_nodes[0]
                self.assertTrue(sdpa.kwargs.get("enable_gqa"))
                self.assertEqual(sdpa.kwargs.get("scale"), 0.25)
                for node, original in zip(sdpa.args[:3], inputs[:3]):
                    padded_shape = (1,) * (4 - rank) + tuple(original.shape)
                    self.assertEqual(tuple(node.meta["val"].shape), padded_shape)
                self.assertFalse(
                    _has_op(result.graph_module, exir_ops.edge.aten.expand_copy.default)
                )

    def test_sdpa_input_ranks(self):
        class SDPA(nn.Module):
            def __init__(self, causal):
                super().__init__()
                self.causal = causal

            def forward(self, q, k, v, mask=None):
                return nn.functional.scaled_dot_product_attention(
                    q, k, v, mask, is_causal=self.causal, scale=0.25
                )

        for rank, mode in ((2, "boolean"), (3, "additive"), (4, "causal")):
            with self.subTest(rank=rank, mode=mode):
                prefix = {2: (), 3: (2,), 4: (2, 2)}[rank]
                mask = None
                if mode == "boolean":
                    mask = torch.ones(3, 5, dtype=torch.bool)
                    mask[:, -1] = False
                elif mode == "additive":
                    mask = torch.linspace(-1, 1, 15).reshape(3, 5)
                inputs = (
                    torch.randn(*prefix, 3, 8),
                    torch.randn(*prefix, 5, 8),
                    torch.randn(*prefix, 5, 8),
                )
                if mask is not None:
                    inputs += (mask,)
                result = self._run_pipeline(SDPA(mode == "causal"), inputs, "SdpaNode")
                nodes = _find_nodes(
                    result.graph_module,
                    exir_ops.edge.aten.scaled_dot_product_attention.default,
                )
                self.assertEqual(len(nodes), 1)
                sdpa = nodes[0]
                for node, original in zip(sdpa.args[:3], inputs[:3]):
                    self.assertEqual(
                        tuple(node.meta["val"].shape),
                        (1,) * (4 - rank) + tuple(original.shape),
                    )
                    if rank == 4:
                        self.assertEqual(node.op, "placeholder")
                self.assertEqual(sdpa.kwargs.get("scale"), 0.25)
                self.assertEqual(len(sdpa.meta["val"].shape), 4)
                if mask is not None:
                    self.assertEqual(
                        tuple(sdpa.args[3].meta["val"].shape), (1, 1, 3, 5)
                    )
                    self.assertEqual(sdpa.args[3].meta["val"].dtype, mask.dtype)
                else:
                    self.assertTrue(sdpa.args[5])
                    self.assertFalse(
                        _has_op(
                            result.graph_module, exir_ops.edge.aten.squeeze_copy.dims
                        )
                    )

    def test_weighted_and_unweighted_rms_norm(self):
        class RMSNorm(nn.Module):
            def __init__(self, weighted):
                super().__init__()
                self.weight = (
                    nn.Parameter(torch.linspace(0.5, 1.5, 16)) if weighted else None
                )

            def forward(self, x):
                norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                return norm if self.weight is None else self.weight * norm

        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                result = self._run_pipeline(
                    RMSNorm(weighted), (torch.randn(2, 3, 16),), "RMSNormNode"
                )
                nodes = _find_nodes(
                    result.graph_module, torch.ops.aten.rms_norm.default
                )
                self.assertEqual(len(nodes), 1)
                self.assertEqual(nodes[0].args[2] is not None, weighted)
                self.assertFalse(
                    _has_op(result.graph_module, exir_ops.edge.aten.rsqrt.default)
                )

    def test_mixed_precision_rms_norm_folds_casts(self):
        class RMSNorm(nn.Module):
            def __init__(self, round_before_weight):
                super().__init__()
                self.weight = nn.Parameter(torch.linspace(0.501, 1.499, 16))
                self.round_before_weight = round_before_weight

            def forward(self, x):
                y = x.float()
                norm = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
                if self.round_before_weight:
                    return norm.to(x.dtype).float() * self.weight
                return (norm * self.weight).to(x.dtype)

        for dtype in (torch.float16, torch.bfloat16):
            for round_before_weight in (False, True):
                with self.subTest(dtype=dtype, round_before_weight=round_before_weight):
                    x = torch.linspace(-2.3, 1.7, 96).reshape(2, 3, 16).to(dtype)
                    model = RMSNorm(round_before_weight)
                    result = self._run_pipeline(
                        model,
                        (x,),
                        "RMSNormNode",
                        rtol=2 * torch.finfo(dtype).eps,
                        atol=2 * torch.finfo(dtype).eps,
                    )
                    nodes = _find_nodes(
                        result.graph_module, torch.ops.aten.rms_norm.default
                    )
                    self.assertEqual(len(nodes), 1)
                    norm = nodes[0]
                    self.assertEqual(norm.args[0].meta["val"].dtype, dtype)
                    self.assertEqual(norm.meta["val"].dtype, dtype)
                    self.assertEqual(norm.args[2] is None, round_before_weight)
                    if norm.args[2] is not None:
                        self.assertEqual(norm.args[2].meta["val"].dtype, dtype)
                    casts = _find_nodes(
                        result.graph_module, exir_ops.edge.aten._to_copy.default
                    )
                    self.assertFalse(any(n.kwargs.get("dtype") == dtype for n in casts))
                    # Check the opted-in fused policy exactly, including rounded weights.
                    weight = None if round_before_weight else model.weight.to(dtype)
                    expected = nn.functional.rms_norm(x, (16,), weight, 1e-6)
                    if round_before_weight:
                        expected = expected.float() * model.weight
                    torch.testing.assert_close(
                        result.module()(x), expected, rtol=0, atol=0
                    )


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
