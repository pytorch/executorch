# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import unittest
from typing import cast

import torch
from executorch.backends.test.graph_builder import GraphBuilder, single_op_builder
from executorch.backends.transforms.fuse_cascaded_transpose_or_permute_ops import (
    FuseCascadedTransposeOrPermuteOps,
)
from executorch.backends.transforms.fuse_cascaded_view_ops import FuseCascadedViewOps
from executorch.backends.transforms.fuse_transpose_or_permute_op_pairs_pass import (
    FuseTransposeOrPermuteOpPairsPass,
)
from executorch.backends.transforms.postpone_permute_below_squeeze_view import (
    PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView,
)
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.backends.transforms.replace_nop_transpose_or_permute_with_view import (
    ReplaceNopTransposeOrPermuteWithViewPass,
)
from executorch.backends.transforms.replace_squeeze_unsqueeze_with_view import (
    ReplaceSqueezeAndUnsqueezeWithViewPass,
)

from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.utils import _pytree as pytree


def count_node(graph_module: torch.fx.GraphModule, target: torch.fx.node.Target) -> int:
    """Count the number of nodes with target `target` in the graph."""
    total = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == target:
            total += 1
    return total


def validate_numerics(
    original: torch.fx.GraphModule,
    modified: torch.fx.GraphModule,
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    pass_name: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Validate that two graph modules produce numerically equivalent outputs."""
    original.eval()
    modified.eval()
    with torch.no_grad():
        orig_out = original(*inputs)
        mod_out = modified(*inputs)

    flat_orig_out, _ = pytree.tree_flatten(orig_out)
    flat_mod_out, _ = pytree.tree_flatten(mod_out)

    for i, (orig_tensor, mod_tensor) in enumerate(zip(flat_orig_out, flat_mod_out)):
        if not torch.allclose(orig_tensor, mod_tensor, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(orig_tensor - mod_tensor)).item()
            raise AssertionError(
                f"Pass validation failed for pass {pass_name}. "
                f"Output tensor {i} differs by max {max_diff:.6e}. "
                f"Expected rtol={rtol}, atol={atol}."
            )


def get_compute_nodes(
    graph_module: torch.fx.GraphModule,
) -> list:
    """Return the target of each call_function node in order."""
    return [
        n.target
        for n in graph_module.graph.nodes
        if n.op == "call_function"
        and n.target
        not in (
            torch.ops.aten.sym_size.int,
            torch.ops.aten.sym_stride.int,
            torch.ops.aten.sym_numel.default,
        )
    ]


# ──────────────────────────────────────────────────────────────────────
# Tests for FuseCascadedTransposeOrPermuteOps
# ──────────────────────────────────────────────────────────────────────


class FuseCascadedTransposeOrPermuteOpsTest(unittest.TestCase):
    def test_structural_permute_composition_preserves_provenance(self) -> None:
        for second_target, expected_target in (
            (
                exir_ops.edge.channels_last.permute_copy.default,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            (
                exir_ops.edge.aten.permute_copy.default,
                exir_ops.edge.aten.permute_copy.default,
            ),
        ):
            with self.subTest(second_target=second_target):
                builder = GraphBuilder()
                x_data = torch.randn(1, 2, 3, 4)
                x = builder.placeholder("x", x_data)
                first = builder.call_operator(
                    op=exir_ops.edge.channels_last.permute_copy.default,
                    args=(x, [0, 2, 3, 1]),
                )
                second = builder.call_operator(
                    op=second_target,
                    args=(first, [0, 1, 3, 2]),
                )
                builder.output([second])
                graph_module = builder.get_graph_module()
                before = copy.deepcopy(graph_module)

                result = cast(
                    PassResult,
                    FuseCascadedTransposeOrPermuteOps()(graph_module),
                )

                self.assertTrue(result.modified)
                self.assertEqual(count_node(result.graph_module, expected_target), 1)
                validate_numerics(
                    before,
                    result.graph_module,
                    [x_data],
                    "FuseCascadedTransposeOrPermuteOps",
                )

    def test_channels_last_input_normalization_pair_is_preserved(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)
        x = builder.placeholder("x", x_data)
        first = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        second = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(first, [0, 3, 1, 2]),
        )
        builder.output([second])
        graph_module = builder.get_graph_module()

        result = cast(
            PassResult,
            FuseCascadedTransposeOrPermuteOps()(graph_module),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            1,
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default),
            1,
        )

    def test_ordinary_mixed_inverse_permutes_are_fused(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 2, 3, 4))
        first = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        second = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(first, [0, 3, 1, 2]),
        )
        builder.output([second])
        graph_module = builder.get_graph_module()

        result = cast(
            PassResult,
            FuseCascadedTransposeOrPermuteOps()(graph_module),
        )

        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            0,
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default),
            0,
        )

    def test_permute_transpose_fusion(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(3, 1, 3, 1, 4))
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 4, 1, 3])
        )
        transpose = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int, args=(permute, 1, 0)
        )
        builder.output([transpose])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = FuseCascadedTransposeOrPermuteOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        gm = result.graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 1)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.transpose_copy.int), 0)
        validate_numerics(
            gm_before,
            gm,
            [torch.randn(3, 1, 3, 1, 4)],
            "FuseCascadedTransposeOrPermuteOps",
        )

    def test_cascaded_permutes_multiple_users(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 3, 4, 5))
        permute1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 3, 1])
        )
        permute2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(permute1, [0, 3, 1, 2])
        )
        permute3 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(permute1, [0, 2, 1, 3])
        )
        permute4 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(permute1, [3, 2, 0, 1])
        )
        permute5 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(permute4, [0, 1, 3, 2])
        )
        builder.output([permute2, permute3, permute5])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = FuseCascadedTransposeOrPermuteOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        validate_numerics(
            gm_before,
            result.graph_module,
            [torch.randn(2, 3, 4, 5)],
            "FuseCascadedTransposeOrPermuteOps",
        )

    def test_permute_view_permute_fuse(self) -> None:
        """permute_3D([0,2,1]) → view(unsqueeze) → permute_4D([0,2,3,1]) should
        be replaced with a single view_copy (permutations cancel out)."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 40, 18)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        v = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 18, 1, 40])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = FuseCascadedTransposeOrPermuteOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        gm = result.graph_module

        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 0)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 1)
        validate_numerics(
            gm_before,
            gm,
            [x_data],
            "FuseCascadedAcrossView",
        )

    def test_permute_view_squeeze_permute_fuse(self) -> None:
        """permute_4D → view(squeeze) → permute_3D should fuse when
        the combined permutation is identity."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 1, 40, 18)
        x = builder.placeholder("x", x_data)
        # NHWC-like permute
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        # Squeeze dim 2 (size 1)
        v = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 18, 40])
        )
        # Inverse 3D permute
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = FuseCascadedTransposeOrPermuteOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        gm = result.graph_module

        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 0)
        validate_numerics(
            gm_before,
            gm,
            [x_data],
            "FuseCascadedSqueezeView",
        )

    def test_transpose_view_permute_fuse(self) -> None:
        """transpose → view(unsqueeze) → permute should fuse when combined
        permutations cancel out."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 40, 18)
        x = builder.placeholder("x", x_data)
        t1 = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int, args=(x, 1, 2)
        )
        v = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(t1, [1, 18, 1, 40])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = FuseCascadedTransposeOrPermuteOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        gm = result.graph_module

        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 0)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.transpose_copy.int), 0)
        validate_numerics(
            gm_before,
            gm,
            [x_data],
            "FuseTransposeViewPermute",
        )

    def test_no_fuse_non_squeeze_view(self) -> None:
        """permute → view (not squeeze/unsqueeze, changes shape) → permute
        should NOT fuse."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 6, 8)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        # This view reshapes 8x6 → 4x12, NOT a squeeze/unsqueeze
        v = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 4, 12])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()

        p = FuseCascadedTransposeOrPermuteOps()
        result = cast(PassResult, p(original))
        # The view is not a squeeze/unsqueeze so cross-view fusion should not fire
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )

    def test_no_fuse_non_cancelling_across_view(self) -> None:
        """permute → view(unsqueeze) → permute where combined permutations
        are NOT identity should NOT be fused away."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 40, 18)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        v = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 18, 1, 40])
        )
        # This permute does NOT cancel with p1
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 1, 3, 2])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = FuseCascadedTransposeOrPermuteOps()
        result = cast(PassResult, p(original))
        # Should NOT have removed both permutes
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "FuseNonCancellingAcrossView",
        )


# ──────────────────────────────────────────────────────────────────────
# Tests for FuseCascadedViewOps
# ──────────────────────────────────────────────────────────────────────


class FuseCascadedViewOpsTest(unittest.TestCase):
    def test_view_fusion(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(8, 5, 3))
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(x, [1, 8, 15])
        )
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(v1, [1, 1, 120])
        )
        v3 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(v2, [120])
        )
        builder.output([v3])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = FuseCascadedViewOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        gm = result.graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 1)
        validate_numerics(
            gm_before,
            gm,
            [torch.randn(8, 5, 3)],
            "FuseCascadedViewOps",
        )

    def test_view_fusion_branched(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(8, 5, 3))
        y = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(x, [1, 8, 15])
        )
        branch1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(y, [1, 1, 120])
        )
        branch2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(y, [120, 1, 1])
        )
        builder.output([branch1, branch2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = FuseCascadedViewOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        gm = result.graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 2)
        validate_numerics(
            gm_before,
            gm,
            [torch.randn(8, 5, 3)],
            "FuseCascadedViewOps",
        )


# ──────────────────────────────────────────────────────────────────────
# Tests for PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView
# ──────────────────────────────────────────────────────────────────────


class PostponePermuteBelowSqueezeViewTest(unittest.TestCase):
    def test_permute3_view4_chains(self) -> None:
        """view→permute→view→permute reordered to view→view→permute→permute."""
        builder = GraphBuilder()
        x_data = torch.randn(3, 1, 768)
        x = builder.placeholder("x", x_data)
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(x, [3, 12, 64])
        )
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v1, [1, 0, 2])
        )
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 12, 3, 64])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v2, [0, 1, 3, 2])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        pass_instance = PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView()
        result = cast(PassResult, pass_instance.call(original))
        self.assertTrue(result.modified)
        gm = result.graph_module
        gm.graph.eliminate_dead_code()

        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 2)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 2)
        targets = get_compute_nodes(gm)
        view_indices = [
            i
            for i, t in enumerate(targets)
            if t == exir_ops.edge.aten.view_copy.default
        ]
        permute_indices = [
            i
            for i, t in enumerate(targets)
            if t == exir_ops.edge.aten.permute_copy.default
        ]
        self.assertTrue(all(v < p for v in view_indices for p in permute_indices))

        validate_numerics(
            gm_before,
            gm,
            [x_data],
            "PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView",
        )

    def test_permute4_view3_chains(self) -> None:
        """4d→permute→view→3d→permute reordered to view→view→permute→permute."""
        builder = GraphBuilder()
        x_data = torch.randn(3, 1, 768)
        x = builder.placeholder("x", x_data)
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(x, [1, 3, 12, 64])
        )
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v1, [3, 1, 0, 2])
        )
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [64, 3, 12])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v2, [2, 1, 0])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        pass_instance = PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView()
        result = cast(PassResult, pass_instance.call(original))
        self.assertTrue(result.modified)
        gm = result.graph_module

        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 2)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 2)
        targets = get_compute_nodes(gm)
        view_indices = [
            i
            for i, t in enumerate(targets)
            if t == exir_ops.edge.aten.view_copy.default
        ]
        permute_indices = [
            i
            for i, t in enumerate(targets)
            if t == exir_ops.edge.aten.permute_copy.default
        ]
        self.assertTrue(all(v < p for v in view_indices for p in permute_indices))

        validate_numerics(
            gm_before,
            gm,
            [x_data],
            "PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView",
        )

    def test_postpone_permute_with_symbolic_shapes(self) -> None:
        class DynamicPermuteViewModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x.view(x.shape[0], 12, 64)
                y = y.permute(1, 0, 2)
                y = y.view(1, 12, x.shape[0], 64)
                return y.permute(0, 1, 3, 2)

        exported_program = torch.export.export(
            DynamicPermuteViewModule(),
            (torch.randn(3, 1, 768),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=8)}},
        )
        edge_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        graph_module = edge_program.exported_program().graph_module

        result = cast(
            PassResult,
            PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView().call(graph_module),
        )

        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.view_copy.default), 2
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )

    def test_negative_not_squeeze_like(self) -> None:
        """View that reshapes (not just squeeze/unsqueeze) should NOT be reordered."""
        builder = GraphBuilder()
        x_data = torch.randn(3, 1, 768)
        x = builder.placeholder("x", x_data)
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(x, [1, 3, 12, 64])
        )
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v1, [3, 1, 0, 2])
        )
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [64, 6, 6])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v2, [2, 1, 0])
        )
        builder.output([p2])
        original = builder.get_graph_module()

        pass_instance = PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView()
        result = cast(PassResult, pass_instance.call(original))
        self.assertFalse(result.modified)

        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.view_copy.default), 2
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default),
            2,
        )
        targets = get_compute_nodes(result.graph_module)
        self.assertEqual(targets[0], exir_ops.edge.aten.view_copy.default)
        self.assertEqual(targets[1], exir_ops.edge.aten.permute_copy.default)


class FuseTransposeOrPermuteOpPairsTest(unittest.TestCase):
    def test_channels_last_input_normalization_pair_is_preserved(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)
        x = builder.placeholder("x", x_data)
        to_nhwc = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        quantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(to_nhwc, 0.25, 0, -128, 127, torch.int8),
        )
        to_nchw = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(quantize, [0, 3, 1, 2]),
        )
        builder.output([to_nchw])
        graph_module = builder.get_graph_module()

        result = cast(PassResult, FuseTransposeOrPermuteOpPairsPass()(graph_module))

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            1,
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default),
            1,
        )

    def test_per_tensor_qdq_is_bypassed(self) -> None:
        for op, x_data in (
            (
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                torch.randn(1, 2, 3, 4),
            ),
            (
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                torch.randint(-128, 127, (1, 2, 3, 4), dtype=torch.int8),
            ),
        ):
            with self.subTest(op=op):
                builder = GraphBuilder()
                x = builder.placeholder("x", x_data)
                to_nhwc = builder.call_operator(
                    op=exir_ops.edge.aten.permute_copy.default,
                    args=(x, [0, 2, 3, 1]),
                )
                qdq = builder.call_operator(
                    op=op,
                    args=(to_nhwc, 0.25, 0, -128, 127, torch.int8),
                )
                to_nchw = builder.call_operator(
                    op=exir_ops.edge.aten.permute_copy.default,
                    args=(qdq, [0, 3, 1, 2]),
                )
                builder.output([to_nchw])
                graph_module = builder.get_graph_module()
                before = copy.deepcopy(graph_module)

                result = cast(
                    PassResult, FuseTransposeOrPermuteOpPairsPass()(graph_module)
                )

                self.assertTrue(result.modified)
                self.assertEqual(
                    count_node(
                        result.graph_module, exir_ops.edge.aten.permute_copy.default
                    ),
                    0,
                )
                validate_numerics(
                    before,
                    result.graph_module,
                    [x_data],
                    "FuseTransposeOrPermuteOpPairsPass",
                )

    def test_per_channel_qdq_is_not_bypassed_without_axis_remap(self) -> None:
        for op, x_data in (
            (
                exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
                torch.randn(1, 2, 3, 4),
            ),
            (
                exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
                torch.randint(-128, 127, (1, 2, 3, 4), dtype=torch.int8),
            ),
        ):
            with self.subTest(op=op):
                builder = GraphBuilder()
                x = builder.placeholder("x", x_data)
                scales = builder.placeholder("scales", torch.tensor([0.25, 0.5]))
                zero_points = builder.placeholder(
                    "zero_points", torch.tensor([0, 0], dtype=torch.int64)
                )
                to_nhwc = builder.call_operator(
                    op=exir_ops.edge.aten.permute_copy.default,
                    args=(x, [0, 2, 3, 1]),
                )
                qdq = builder.call_operator(
                    op=op,
                    # NHWC axis 3 would need to become NCHW axis 1 if the
                    # surrounding permutes were removed.
                    args=(to_nhwc, scales, zero_points, 3, -128, 127, torch.int8),
                )
                to_nchw = builder.call_operator(
                    op=exir_ops.edge.aten.permute_copy.default,
                    args=(qdq, [0, 3, 1, 2]),
                )
                builder.output([to_nchw])
                graph_module = builder.get_graph_module()
                before = copy.deepcopy(graph_module)

                result = cast(
                    PassResult, FuseTransposeOrPermuteOpPairsPass()(graph_module)
                )

                self.assertFalse(result.modified)
                self.assertEqual(
                    count_node(
                        result.graph_module, exir_ops.edge.aten.permute_copy.default
                    ),
                    2,
                )
                validate_numerics(
                    before,
                    result.graph_module,
                    [x_data, torch.tensor([0.25, 0.5]), torch.tensor([0, 0])],
                    "FuseTransposeOrPermuteOpPairsPass",
                )

    def test_per_channel_qdq_chain_is_not_bypassed(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4)
        scales_data = torch.tensor([0.25, 0.5])
        zero_points_data = torch.tensor([0, 0], dtype=torch.int64)
        x = builder.placeholder("x", x_data)
        scales = builder.placeholder("scales", scales_data)
        zero_points = builder.placeholder("zero_points", zero_points_data)
        to_nhwc = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        quantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
            args=(to_nhwc, scales, zero_points, 3, -128, 127, torch.int8),
        )
        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
            args=(quantize, scales, zero_points, 3, -128, 127, torch.int8),
        )
        to_nchw = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(dequantize, [0, 3, 1, 2]),
        )
        builder.output([to_nchw])
        graph_module = builder.get_graph_module()
        before = copy.deepcopy(graph_module)

        result = cast(PassResult, FuseTransposeOrPermuteOpPairsPass()(graph_module))

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default),
            2,
        )
        validate_numerics(
            before,
            result.graph_module,
            [x_data, scales_data, zero_points_data],
            "FuseTransposeOrPermuteOpPairsPass",
        )

    def test_per_channel_branch_blocks_shared_permute_fusion(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4)
        scales_data = torch.tensor([0.25, 0.5])
        zero_points_data = torch.tensor([0, 0], dtype=torch.int64)
        x = builder.placeholder("x", x_data)
        scales = builder.placeholder("scales", scales_data)
        zero_points = builder.placeholder("zero_points", zero_points_data)
        to_nhwc = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        per_tensor = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(to_nhwc, 0.25, 0, -128, 127, torch.int8),
        )
        per_channel = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
            args=(to_nhwc, scales, zero_points, 3, -128, 127, torch.int8),
        )
        per_tensor_out = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(per_tensor, [0, 3, 1, 2]),
        )
        per_channel_out = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(per_channel, [0, 3, 1, 2]),
        )
        builder.output([per_tensor_out, per_channel_out])
        graph_module = builder.get_graph_module()

        result = cast(PassResult, FuseTransposeOrPermuteOpPairsPass()(graph_module))

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default),
            3,
        )


# ──────────────────────────────────────────────────────────────────────
# Tests for structural layout boundary propagation
# ──────────────────────────────────────────────────────────────────────


class StructuralLayoutBoundaryPropagationTest(unittest.TestCase):
    @staticmethod
    def _layout_add_graph(
        bias_name: str, bias_data: torch.Tensor
    ) -> tuple[torch.fx.GraphModule, torch.Tensor]:
        builder = GraphBuilder()
        x_data = torch.randn(1, 8, 8, 4)
        x = builder.placeholder("x", x_data)
        bias = builder.placeholder(bias_name, bias_data)
        to_nchw = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 3, 1, 2]),
        )
        add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(to_nchw, bias),
        )
        to_nhwc = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(add, [0, 2, 3, 1]),
        )
        builder.output([to_nhwc])
        return builder.get_graph_module(), x_data

    @staticmethod
    def _layout_pad_graph(
        shape: tuple[int, ...],
        to_inner: list[int],
        to_outer: list[int],
        pad: list[int],
    ) -> tuple[torch.fx.GraphModule, torch.Tensor]:
        builder = GraphBuilder()
        x_data = torch.randn(*shape)
        x = builder.placeholder("x", x_data)
        inner = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, to_inner),
        )
        padded = builder.call_operator(
            op=exir_ops.edge.aten.constant_pad_nd.default,
            args=(inner, pad, 0.0),
        )
        outer = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(padded, to_outer),
        )
        builder.output([outer])
        return builder.get_graph_module(), x_data

    def test_layout_pad_retarget_is_opt_in(self) -> None:
        graph_module, x_data = self._layout_pad_graph(
            (1, 8, 8, 3),
            [0, 3, 1, 2],
            [0, 2, 3, 1],
            [0, 0, 0, 0, 0, 1],
        )
        before = copy.deepcopy(graph_module)

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(
                allow_layout_boundary_propagation=True,
            )(graph_module),
        )

        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.constant_pad_nd.default),
            1,
        )
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.constant_pad_nd.default,
            ),
            0,
        )
        validate_numerics(
            before,
            result.graph_module,
            [x_data],
            "RemovePermutesAroundElementwiseOps",
        )

    def test_layout_pad_retarget_requires_rank_four(self) -> None:
        for shape, to_inner, to_outer, pad, expected_target in (
            (
                (1, 8, 8, 3),
                [0, 3, 1, 2],
                [0, 2, 3, 1],
                [0, 0, 0, 0, 0, 1],
                exir_ops.edge.channels_last.constant_pad_nd.default,
            ),
            (
                (2, 8, 3),
                [0, 2, 1],
                [0, 2, 1],
                [0, 0, 0, 1],
                exir_ops.edge.aten.constant_pad_nd.default,
            ),
        ):
            with self.subTest(shape=shape):
                graph_module, x_data = self._layout_pad_graph(
                    shape, to_inner, to_outer, pad
                )
                before = copy.deepcopy(graph_module)

                result = cast(
                    PassResult,
                    RemovePermutesAroundElementwiseOps(
                        allow_layout_boundary_propagation=True,
                        layout_pad_target=(
                            exir_ops.edge.channels_last.constant_pad_nd.default
                        ),
                    )(graph_module),
                )

                self.assertTrue(result.modified)
                self.assertEqual(count_node(result.graph_module, expected_target), 1)
                validate_numerics(
                    before,
                    result.graph_module,
                    [x_data],
                    "RemovePermutesAroundElementwiseOps",
                )

    def test_existing_layout_pad_is_remapped(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 3, 8, 8))
        pad = builder.call_operator(
            op=exir_ops.edge.channels_last.constant_pad_nd.default,
            args=(x, [0, 0, 0, 0, 0, 1], 0.0),
        )
        builder.output([pad])

        RemovePermutesAroundElementwiseOps().update_pad(
            pad.node,
            [0, 3, 1, 2],
            layout_region=True,
        )

        self.assertEqual(pad.node.args[1], [0, 1])

    def test_pair_fusion_recognizes_structural_permutes(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4)
        x = builder.placeholder("x", x_data)
        to_nhwc = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        quantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(to_nhwc, 0.25, 0, -128, 127, torch.int8),
        )
        to_nchw = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(quantize, [0, 3, 1, 2]),
        )
        builder.output([to_nchw])
        graph_module = builder.get_graph_module()
        before = copy.deepcopy(graph_module)

        result = cast(PassResult, FuseTransposeOrPermuteOpPairsPass()(graph_module))

        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            0,
        )
        validate_numerics(
            before,
            result.graph_module,
            [x_data],
            "FuseTransposeOrPermuteOpPairsPass",
        )

    def test_pair_fusion_respects_backend_propagation_barrier(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4)
        x = builder.placeholder("x", x_data)
        to_nhwc = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        quantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(to_nhwc, 0.25, 0, -128, 127, torch.int8),
        )
        to_nchw = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(quantize, [0, 3, 1, 2]),
        )
        builder.output([to_nchw])
        graph_module = builder.get_graph_module()

        result = cast(
            PassResult,
            FuseTransposeOrPermuteOpPairsPass(
                can_propagate=lambda node: node.target
                != exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            )(graph_module),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            2,
        )

    def test_pair_fusion_does_not_bypass_structural_per_channel_qdq(self) -> None:
        for op, x_data in (
            (
                exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
                torch.randn(1, 2, 3, 4),
            ),
            (
                exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
                torch.randint(-128, 127, (1, 2, 3, 4), dtype=torch.int8),
            ),
        ):
            with self.subTest(op=op):
                builder = GraphBuilder()
                x = builder.placeholder("x", x_data)
                scales = builder.placeholder("scales", torch.tensor([0.25, 0.5]))
                zero_points = builder.placeholder(
                    "zero_points", torch.tensor([0, 0], dtype=torch.int64)
                )
                to_nhwc = builder.call_operator(
                    op=exir_ops.edge.channels_last.permute_copy.default,
                    args=(x, [0, 2, 3, 1]),
                )
                qdq = builder.call_operator(
                    op=op,
                    args=(to_nhwc, scales, zero_points, 3, -128, 127, torch.int8),
                )
                to_nchw = builder.call_operator(
                    op=exir_ops.edge.channels_last.permute_copy.default,
                    args=(qdq, [0, 3, 1, 2]),
                )
                builder.output([to_nchw])
                graph_module = builder.get_graph_module()

                result = cast(
                    PassResult, FuseTransposeOrPermuteOpPairsPass()(graph_module)
                )

                self.assertFalse(result.modified)
                self.assertEqual(
                    count_node(
                        result.graph_module,
                        exir_ops.edge.channels_last.permute_copy.default,
                    ),
                    2,
                )

    def test_layout_copy_moves_to_static_output_boundary(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4)
        x = builder.placeholder("x", x_data)
        permute = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        output = builder.call_operator(
            op=exir_ops.edge.aten.hardtanh.default,
            args=(permute,),
        )
        builder.output([output])
        graph_module = builder.get_graph_module()
        before = copy.deepcopy(graph_module)

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(allow_layout_boundary_propagation=True)(
                graph_module
            ),
        )

        self.assertTrue(result.modified)
        surviving_permute = result.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.channels_last.permute_copy.default,
        )[0]
        self.assertEqual(
            surviving_permute.args[0].target,
            exir_ops.edge.aten.hardtanh.default,
        )
        validate_numerics(
            before,
            result.graph_module,
            [x_data],
            "RemovePermutesAroundElementwiseOps",
        )

    def test_layout_region_terminates_at_backend_barrier(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4)
        x = builder.placeholder("x", x_data)
        to_nhwc = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        activation = builder.call_operator(
            op=exir_ops.edge.aten.hardtanh.default,
            args=(to_nhwc,),
        )
        to_nchw = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(activation, [0, 3, 1, 2]),
        )
        barrier = builder.call_operator(
            op=exir_ops.edge.aten.relu.default,
            args=(activation,),
        )
        builder.output([to_nchw, barrier])
        graph_module = builder.get_graph_module()
        before = copy.deepcopy(graph_module)

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(
                allow_layout_boundary_propagation=True,
                can_propagate=lambda node: node.target
                != exir_ops.edge.aten.relu.default,
            )(graph_module),
        )

        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            1,
        )
        validate_numerics(
            before,
            result.graph_module,
            [x_data],
            "RemovePermutesAroundElementwiseOps",
        )

    def test_layout_copy_does_not_fork_to_more_boundaries(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 2, 3, 4))
        permute = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        first = builder.call_operator(
            op=exir_ops.edge.aten.hardtanh.default,
            args=(permute,),
        )
        second = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor,
            args=(permute, permute),
        )
        builder.output([first, second])
        graph_module = builder.get_graph_module()

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(allow_layout_boundary_propagation=True)(
                graph_module
            ),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            1,
        )

    def test_shared_incoming_layout_copy_is_not_credited_as_removed(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 2, 3, 4))
        permute = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        main = builder.call_operator(
            op=exir_ops.edge.aten.hardtanh.default,
            args=(permute,),
        )
        auxiliary = builder.call_operator(
            op=exir_ops.edge.aten._softmax.default,
            args=(permute, -1, False),
        )
        builder.output([main, auxiliary])
        graph_module = builder.get_graph_module()

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(allow_layout_boundary_propagation=True)(
                graph_module
            ),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            1,
        )

    def test_input_boundary_does_not_alias_permuted_and_unpermuted_input(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 2, 2)
        x = builder.placeholder("x", x_data)
        permute = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(permute, x),
        )
        builder.output([add])
        graph_module = builder.get_graph_module()
        before = copy.deepcopy(graph_module)

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(allow_layout_boundary_propagation=True)(
                graph_module
            ),
        )

        self.assertFalse(result.modified)
        validate_numerics(
            before,
            result.graph_module,
            [x_data],
            "RemovePermutesAroundElementwiseOps",
        )

    def test_output_boundary_does_not_alias_two_output_edges(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 2, 3, 4)
        x = builder.placeholder("x", x_data)
        to_nhwc = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        activation = builder.call_operator(
            op=exir_ops.edge.aten.hardtanh.default,
            args=(to_nhwc,),
        )
        to_nchw = builder.call_operator(
            op=exir_ops.edge.channels_last.permute_copy.default,
            args=(activation, [0, 3, 1, 2]),
        )
        builder.output([to_nchw, activation])
        graph_module = builder.get_graph_module()
        before = copy.deepcopy(graph_module)

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(allow_layout_boundary_propagation=True)(
                graph_module
            ),
        )

        self.assertFalse(result.modified)
        validate_numerics(
            before,
            result.graph_module,
            [x_data],
            "RemovePermutesAroundElementwiseOps",
        )

    def test_layout_copy_does_not_cross_unknown_cost_boundary(self) -> None:
        class DynamicDequantize(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                permuted = torch.ops.channels_last.permute_copy(x, [0, 2, 3, 1])
                return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    permuted, 0.1, 0, -128, 127, torch.int8
                )

        inputs = (torch.randint(-128, 127, (1, 4, 8, 10), dtype=torch.int8),)
        exported = torch.export.export(
            DynamicDequantize(),
            inputs,
            dynamic_shapes={"x": {2: torch.export.Dim("height", min=2, max=16)}},
        )
        edge = to_edge(
            exported,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )
        graph_module = edge.exported_program().graph_module

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(
                exported_program=edge.exported_program(),
                allow_layout_boundary_propagation=True,
            )(graph_module),
        )

        self.assertFalse(result.modified)
        permute = result.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.channels_last.permute_copy.default,
        )[0]
        dequantize = result.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )[0]
        self.assertIs(dequantize.args[0], permute)
        self.assertEqual(permute.meta["val"].dtype, torch.int8)

    def test_layout_copy_rejects_rank_mismatched_runtime_input(self) -> None:
        bias_data = torch.randn(4, 1, 1)
        graph_module, x_data = self._layout_add_graph("bias", bias_data)
        before = copy.deepcopy(graph_module)

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(allow_layout_boundary_propagation=True)(
                graph_module
            ),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            2,
        )
        validate_numerics(
            before,
            result.graph_module,
            [x_data, bias_data],
            "RemovePermutesAroundElementwiseOps",
        )

    def test_layout_copy_rejects_spatial_constant_reordering(self) -> None:
        bias_data = torch.randn(4, 8, 8)
        graph_module, x_data = self._layout_add_graph("b_bias", bias_data)
        before = copy.deepcopy(graph_module)

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(allow_layout_boundary_propagation=True)(
                graph_module
            ),
        )

        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            2,
        )
        validate_numerics(
            before,
            result.graph_module,
            [x_data, bias_data],
            "RemovePermutesAroundElementwiseOps",
        )

    def test_layout_copy_reshapes_channel_constant_without_copy(self) -> None:
        bias_data = torch.randn(4, 1, 1)
        graph_module, x_data = self._layout_add_graph("b_bias", bias_data)
        before = copy.deepcopy(graph_module)

        result = cast(
            PassResult,
            RemovePermutesAroundElementwiseOps(allow_layout_boundary_propagation=True)(
                graph_module
            ),
        )

        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(
                result.graph_module,
                exir_ops.edge.channels_last.permute_copy.default,
            ),
            0,
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.view_copy.default),
            1,
        )
        validate_numerics(
            before,
            result.graph_module,
            [x_data, bias_data],
            "RemovePermutesAroundElementwiseOps",
        )


# ─────────────────────────────────────
# Tests for ReplaceNopTransposeOrPermuteWithViewPass
# ─────────────────────────────────────


class ReplaceNopTransposeOrPermuteWithViewTest(unittest.TestCase):
    def test_replace_nop_transpose_with_view_float(self) -> None:
        x = torch.randn(2, 1, 3, 1)
        gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(x, 1, 3),
        )
        gm_before = copy.deepcopy(gm)

        p = ReplaceNopTransposeOrPermuteWithViewPass()
        result = cast(PassResult, p(gm))
        self.assertTrue(result.modified)
        gm_after = result.graph_module
        self.assertEqual(
            count_node(gm_after, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(count_node(gm_after, exir_ops.edge.aten.view_copy.default), 1)
        validate_numerics(
            gm_before, gm_after, [x], "ReplaceNopTransposeOrPermuteWithViewPass"
        )

    def test_replace_nop_transpose_with_view_int(self) -> None:
        x = torch.randint(low=0, high=100, size=(2, 1, 5), dtype=torch.int64)
        gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(x, 1, 0),
        )
        gm_before = copy.deepcopy(gm)

        p = ReplaceNopTransposeOrPermuteWithViewPass()
        result = cast(PassResult, p(gm))
        self.assertTrue(result.modified)
        gm_after = result.graph_module
        self.assertEqual(count_node(gm_after, exir_ops.edge.aten.transpose_copy.int), 0)
        self.assertEqual(count_node(gm_after, exir_ops.edge.aten.view_copy.default), 1)
        validate_numerics(
            gm_before, gm_after, [x], "ReplaceNopTransposeOrPermuteWithViewPass"
        )

    def test_replace_nop_permute_5d(self) -> None:
        x = torch.randn(3, 1, 3, 1, 4)
        gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 2, 4, 1, 3]),
        )
        gm_before = copy.deepcopy(gm)

        p = ReplaceNopTransposeOrPermuteWithViewPass()
        result = cast(PassResult, p(gm))
        self.assertTrue(result.modified)
        gm_after = result.graph_module
        self.assertEqual(
            count_node(gm_after, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(count_node(gm_after, exir_ops.edge.aten.view_copy.default), 1)
        validate_numerics(
            gm_before, gm_after, [x], "ReplaceNopTransposeOrPermuteWithViewPass"
        )

    def test_replace_nop_permute_3d(self) -> None:
        x = torch.randn(1, 3, 4)
        gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [1, 2, 0]),
        )
        gm_before = copy.deepcopy(gm)

        p = ReplaceNopTransposeOrPermuteWithViewPass()
        result = cast(PassResult, p(gm))
        self.assertTrue(result.modified)
        gm_after = result.graph_module
        self.assertEqual(
            count_node(gm_after, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(count_node(gm_after, exir_ops.edge.aten.view_copy.default), 1)
        validate_numerics(
            gm_before, gm_after, [x], "ReplaceNopTransposeOrPermuteWithViewPass"
        )


# ──────────────────────────────────────────────────────────────────────
# Tests for RemovePermutesAroundElementwiseOps cross-view handling
# ──────────────────────────────────────────────────────────────────────


def _canonicalize_and_remove_permutes(
    gm: torch.fx.GraphModule,
) -> PassResult:
    """Canonicalise squeeze/unsqueeze to view_copy, then remove permutes.

    RemovePermutesAroundElementwiseOps reasons about view_copy alone, so this
    mirrors the ordering every backend pipeline uses.
    """
    canonical = cast(
        PassResult, ReplaceSqueezeAndUnsqueezeWithViewPass()(gm)
    ).graph_module
    return cast(PassResult, RemovePermutesAroundElementwiseOps()(canonical))


class RemovePermutesAcrossViewTest(unittest.TestCase):
    def test_permute_view_squeeze_elementwise_view_unsqueeze_permute(self) -> None:
        """permute(3D) → view(unsqueeze) → mul(4D) → view(squeeze) → permute(3D)
        should have both permutes removed."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 128, 16)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 16, 1, 128])
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(v1, v1))
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mul, [1, 16, 128])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v2, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "RemovePermutesAcrossView",
        )

    def test_4d_permute_squeeze_clamp_3d_permute(self) -> None:
        """Cascade detector conv→LN boundary: permute_4D([0,3,1,2]) →
        view(squeeze) → hardtanh → permute_3D([0,2,1]).
        The two permutes should cancel across the squeeze+clamp."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 1, 16, 128)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 128, 16])
        )
        clamp = builder.call_operator(
            op=exir_ops.edge.aten.hardtanh.default, args=(v1,)
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(clamp, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "4D_permute_squeeze_clamp_3D_permute",
        )

    def test_permute_unsqueeze_cat_mul_squeeze_permute(self) -> None:
        """Complex interaction: permute(3D) → view(unsqueeze to 4D) →
        cat(two branches) → mul → view(squeeze to 3D) → permute(3D).
        Tests cat + mul interacting with view/squeeze/unsqueeze boundaries."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 128, 16)
        y_data = torch.randn(1, 128, 16)
        x = builder.placeholder("x", x_data)
        y = builder.placeholder("y", y_data)
        # Permute both inputs
        px = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        py = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(y, [0, 2, 1])
        )
        # Unsqueeze via view to 4D
        vx = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(px, [1, 16, 1, 128])
        )
        vy = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(py, [1, 16, 1, 128])
        )
        # Cat along dim 2 (the unsqueezed dim)
        cat = builder.call_operator(
            op=exir_ops.edge.aten.cat.default, args=([vx, vy], 2)
        )
        # Mul with itself
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(cat, cat))
        # Squeeze back via view
        v_sq = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mul, [1, 16, 256])
        )
        # End permute
        p_end = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v_sq, [0, 2, 1])
        )
        builder.output([p_end])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        # The cat changes output shape so squeeze view won't match the
        # original unsqueeze pattern; the pass should not fire here.
        self.assertFalse(result.modified)
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data, y_data],
            "permute_unsqueeze_cat_mul_squeeze_permute",
        )

    def test_permute_view_add_sub_mul_view_permute(self) -> None:
        """Chain of multiple elementwise ops between view boundaries:
        permute(3D) → view(unsqueeze) → add → sub → mul → view(squeeze) → permute(3D).
        All three elementwise ops should be handled."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 128, 16)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        # Unsqueeze via view
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 16, 1, 128])
        )
        # Chain of elementwise ops
        add = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(v1, v1))
        sub = builder.call_operator(op=exir_ops.edge.aten.sub.Tensor, args=(add, v1))
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(sub, sub))
        # Squeeze via view
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mul, [1, 16, 128])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v2, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "permute_view_add_sub_mul_view_permute",
        )

    def test_permute_constant_pad_nd_permute(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 64, 64, 3)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        pad = builder.call_operator(
            op=exir_ops.edge.aten.constant_pad_nd.default,
            args=(p1, [0, 0, 0, 0, 0, 1], 0.0),
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(pad, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.constant_pad_nd.default),
            1,
        )

        pad_nodes = [
            node
            for node in result.graph_module.graph.nodes
            if node.target == exir_ops.edge.aten.constant_pad_nd.default
        ]
        self.assertEqual(pad_nodes[0].args[1], [0, 1])
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "permute_constant_pad_nd_permute",
        )

    def test_permute_aten_pad_permute(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 64, 64, 3)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        pad = builder.call_operator(
            op=exir_ops.edge.aten.pad.default,
            args=(p1, [0, 0, 0, 0, 0, 1], "constant", 0.0),
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(pad, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.pad.default), 1
        )

        pad_nodes = [
            node
            for node in result.graph_module.graph.nodes
            if node.target == exir_ops.edge.aten.pad.default
        ]
        self.assertEqual(pad_nodes[0].args[1], [0, 1])
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "permute_aten_pad_permute",
        )

    def test_permute_squeeze_clamp_add_permute(self) -> None:
        """4D permute → squeeze(view) → hardtanh → add(with self) → 3D permute.
        Tests clamp + add interacting across a squeeze boundary."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 1, 16, 128)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        # Squeeze dim 2 (size 1) via view
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 128, 16])
        )
        clamp = builder.call_operator(
            op=exir_ops.edge.aten.hardtanh.default, args=(v1,)
        )
        add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(clamp, clamp)
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(add, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "permute_squeeze_clamp_add_permute",
        )

    def test_no_fire_non_squeeze_view(self) -> None:
        """permute → view (not a squeeze/unsqueeze, changes shape) → mul → permute.
        The pass should NOT remove permutes when the view is not a simple
        squeeze/unsqueeze."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 6, 8)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        # This view reshapes 8x6 → 4x12, which is NOT a squeeze/unsqueeze
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 4, 12])
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(v1, v1))
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(mul, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        # Should NOT have removed permutes (view is not squeeze/unsqueeze-like)
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )

    def test_permute_unsqueeze_copy_mul_squeeze_copy_permute(self) -> None:
        """permute(3D) → unsqueeze_copy(dim=2) → mul(4D) → squeeze_copy(dim=2) → permute(3D).
        Canonicalisation rewrites both shape ops to view_copy, which
        _adapt_permute_across_view then crosses."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 128, 16)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        unsq = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(p1, 2)
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(unsq, unsq))
        sq = builder.call_operator(
            op=exir_ops.edge.aten.squeeze_copy.dim, args=(mul, 2)
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(sq, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = _canonicalize_and_remove_permutes(original)
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "permute_unsqueeze_copy_mul_squeeze_copy_permute",
        )

    def test_4d_permute_squeeze_copy_clamp_3d_permute(self) -> None:
        """4D permute([0,3,1,2]) → squeeze_copy(dim=2) → hardtanh → 3D permute([0,2,1]).
        Covers a rank change at the start boundary, reached through the
        view_copy that canonicalisation leaves behind."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 1, 16, 128)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        sq = builder.call_operator(op=exir_ops.edge.aten.squeeze_copy.dim, args=(p1, 2))
        clamp = builder.call_operator(
            op=exir_ops.edge.aten.hardtanh.default, args=(sq,)
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(clamp, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = _canonicalize_and_remove_permutes(original)
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "4D_permute_squeeze_copy_clamp_3D_permute",
        )

    def test_4d_permute_squeeze_view_slice_mul_3d_permute(self) -> None:
        """4D permute([2,0,1,3]) → view(squeeze dim 0) → slice → mul → permute([1,0,2]).
        Regression test for the Transformer pattern where the squeezed dim
        position (0) differs from its permutation value (perm[0]=2).
        Without the fix, _adapt_permute_across_view confuses the position
        with the value, causing the pass to create an invalid subgraph that
        leads to a shape mismatch at runtime."""
        builder = GraphBuilder()
        # Distinct dim sizes to expose mismatched slicing
        x_data = torch.randn(10, 32, 1, 64)
        x = builder.placeholder("x", x_data)
        # Permute puts the size-1 dim (input dim 2) at position 0
        # [10, 32, 1, 64] -> [1, 10, 32, 64]
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [2, 0, 1, 3])
        )
        # Squeeze dim 0 (size 1) via view_copy: [10, 32, 64]
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [10, 32, 64])
        )
        # Slice dim 0, taking 3 elements from size 10
        sl = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor, args=(v1, 0, 0, 3)
        )
        # Elementwise op
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(sl, sl))
        # End permute [1, 0, 2]: swap dims 0 and 1
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(mul, [1, 0, 2])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        # With the fix, the adapted permutation becomes identity [0,1,2],
        # so no matching end permute is found and the graph is unchanged.
        # Before the fix, the wrong adapted permutation [1,0,2] would match
        # the end permute and create an invalid subgraph, causing a crash.
        result = cast(PassResult, p(original))
        self.assertFalse(result.modified)
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "4D_permute_squeeze_view_slice_mul_3D_permute",
        )

    def test_permute_unsqueeze_copy_neg_dim_mul_squeeze_copy_permute(self) -> None:
        """permute(3D) → unsqueeze_copy(dim=-1) → mul(4D) → squeeze_copy(dim=3) → permute(3D).
        Tests unsqueeze with negative dim (output-space rank+1 normalization)
        and dim=rank edge case that would IndexError with incorrect handling."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 128, 16)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        unsq = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(p1, -1)
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(unsq, unsq))
        sq = builder.call_operator(
            op=exir_ops.edge.aten.squeeze_copy.dim, args=(mul, 3)
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(sq, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = _canonicalize_and_remove_permutes(original)
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "permute_unsqueeze_copy_neg_dim_mul_squeeze_copy_permute",
        )

    def test_unsqueeze_at_moved_position(self) -> None:
        """The permutation moves the unsqueeze position (P[index] != index), so
        the adapted permutation must be built from P[index], not index."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 8, 16)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        u = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(p1, 1)
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(u, u))
        sq = builder.call_operator(
            op=exir_ops.edge.aten.squeeze_copy.dim, args=(mul, 1)
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(sq, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = _canonicalize_and_remove_permutes(original)
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "UnsqueezeAtMovedPosition"
        )

    def test_squeeze_dims_multiple_unit_dims(self) -> None:
        """squeeze_copy.dims dropping more than one unit dim at once."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 8, 16)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        u1 = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(p1, 1)
        )
        u2 = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(u1, 2)
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(u2, u2))
        sq = builder.call_operator(
            op=exir_ops.edge.aten.squeeze_copy.dims, args=(mul, [1, 2])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(sq, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = _canonicalize_and_remove_permutes(original)
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "SqueezeDimsMultiple"
        )

    def test_view_copy_multi_unit_dim_rank_change(self) -> None:
        """A view_copy adding and removing two unit dims at once.

        Exercises the N-dim rank change directly, without relying on
        canonicalisation to produce the view_copy.
        """
        builder = GraphBuilder()
        x_data = torch.randn(1, 8, 16)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        v1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(p1, [1, 1, 1, 16, 8])
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(v1, v1))
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mul, [1, 16, 8])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v2, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "ViewCopyMultiUnitDim"
        )

    def test_squeeze_on_non_unit_dim_is_not_optimized(self) -> None:
        """A squeeze only drops size-1 dims, so listing a non-unit dim makes it a
        no-op. Canonicalisation turns it into a same-rank view_copy, which is
        not a crossable rank change, so the region must be skipped. Covers both
        the squeeze_copy.dim and squeeze_copy.dims overloads."""
        for op, dim_arg, name in (
            (exir_ops.edge.aten.squeeze_copy.dim, 1, "SqueezeDimNonUnit"),
            (exir_ops.edge.aten.squeeze_copy.dims, [1], "SqueezeDimsNonUnit"),
        ):
            with self.subTest(name=name):
                builder = GraphBuilder()
                x_data = torch.randn(1, 8, 16)
                x = builder.placeholder("x", x_data)
                p1 = builder.call_operator(
                    op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
                )
                mul = builder.call_operator(
                    op=exir_ops.edge.aten.mul.Tensor, args=(p1, p1)
                )
                # dim 1 has size 16, so the squeeze leaves it in place.
                sq = builder.call_operator(op=op, args=(mul, dim_arg))
                p2 = builder.call_operator(
                    op=exir_ops.edge.aten.permute_copy.default, args=(sq, [0, 2, 1])
                )
                builder.output([p2])
                original = builder.get_graph_module()
                gm_before = copy.deepcopy(original)

                result = _canonicalize_and_remove_permutes(original)
                self.assertFalse(result.modified)
                validate_numerics(gm_before, result.graph_module, [x_data], name)

    def test_upstream_view_rank_mismatch_no_crash(self) -> None:
        """Regression test for IndexError when a squeeze/unsqueeze view_copy
        is reached via upstream traversal with a permutation whose rank does
        not match the view's input tensor rank.

        Graph:
            full([16, 128], 1.0)           x [1, 128, 16]
                    |                            |
            view_copy (unsqueeze 2D→3D)    permute [0, 2, 1]
            [1, 16, 128]                   [1, 16, 128]
                    \\                          /
                     ---- add (3D) -----------
                            |
                      permute [0, 2, 1]
                            |
                         output

        The view_copy (unsqueeze) is reached as an upstream input to `add`.
        Its node_start_permute gets the 3D permutation [0, 2, 1], but its
        input (the full op) is 2D.  Before the fix, update_view_copy would
        crash with IndexError: tuple index out of range."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 128, 16)
        x = builder.placeholder("x", x_data)
        # 2D constant — treated as compile-time constant by _is_constant
        const_2d = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=([16, 128], 1.0)
        )
        # Unsqueeze via view_copy: [16, 128] → [1, 16, 128]
        view_unsq = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(const_2d, [1, 16, 128])
        )
        # Start permute: [1, 128, 16] → [1, 16, 128]
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        # Add the permuted input with the unsqueezed constant
        add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(p1, view_unsq)
        )
        # End permute: [1, 16, 128] → [1, 128, 16]
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(add, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        # Should not crash, and should skip the subgraph due to rank mismatch
        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        # The subgraph is skipped, so the graph should be unmodified
        self.assertFalse(result.modified)
        # Both permutes are preserved
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "upstream_view_rank_mismatch_no_crash",
        )

    def test_permutation_sink_view_splitting_the_non_unit_dim(self) -> None:
        """A reshape whose input has a single non-unit dim is a permutation sink
        even when it splits that dim rather than flattening it:
        (1, 1, 1, 4) -> (1, 2, 2) walks the same elements under any layout, so
        the upstream permute is dropped with no compensating permute and the
        view's shape arg must be left untouched."""
        x_data = torch.randn(1, 4, 1, 1)
        builder = GraphBuilder()
        x = builder.placeholder("x", x_data)
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 3, 1])
        )
        mul = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor, args=(permute, permute)
        )
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mul, [1, 2, 2])
        )
        builder.output([view])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        (view_after,) = result.graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.view_copy.default
        )
        self.assertEqual(view_after.args[1], [1, 2, 2])
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "permutation_sink_view_splitting_the_non_unit_dim",
        )

    def test_upstream_squeeze_view_rank_mismatch_no_crash(self) -> None:
        """Regression test for IndexError when a squeeze view_copy is reached
        via upstream traversal with a permutation at the view's output rank.

        Graph:
            y [8, 4, 1]                    x [4, 8]
                    |                            |
            view_copy (squeeze 3D→2D)      permute [1, 0]
            [8, 4]                         [8, 4]
                    \\                          /
                     ---- add (2D) -----------
                            |
                      permute [1, 0]
                            |
                         output

        `visit` reaches the view_copy from `add` with the 2D permutation
        [1, 0], but the view's input is 3D, so the squeezed position (2) is
        out of range for the permutation. Before the fix,
        _adapt_permute_across_view crashed with IndexError: list index out
        of range."""
        builder = GraphBuilder()
        x_data = torch.randn(4, 8)
        y_data = torch.randn(8, 4, 1)
        x = builder.placeholder("x", x_data)
        y = builder.placeholder("y", y_data)
        # Squeeze via view_copy: [8, 4, 1] → [8, 4]
        view_sq = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(y, [8, 4])
        )
        # Start permute: [4, 8] → [8, 4]
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [1, 0])
        )
        add = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(p1, view_sq)
        )
        # End permute: [8, 4] → [4, 8]
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(add, [1, 0])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        # Should not crash, and should skip the subgraph due to rank mismatch
        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data, y_data],
            "upstream_squeeze_view_rank_mismatch_no_crash",
        )

    def test_broadcast_rank_increase_no_crash(self) -> None:
        """Regression test for IndexError when broadcasting raises a node's
        rank above the rank of the permutation it is visited with.

        Graph:
            x [1, 8]                full([1, 1, 1])
                |                        |
            permute [1, 0]               |
            [8, 1]                       |
                \\                       /
                 ------ add ------------
                    [1, 8, 1]  (rank 3)
                        |
                slice_copy(dim=2)
                        |
                view_copy -> [8]   (permutation sink)

        `add` is visited with the rank-2 permutation [1, 0] but broadcasts to
        rank 3. The numel-1 `full` input and the sink `view_copy` both let
        traversal terminate without ever meeting a rank-matched permute, so the
        subgraph closed with a rank-2 permutation on rank-3 nodes. Before the
        fix, update_slice_copy did `start_permute[2]` and raised
        IndexError: list index out of range."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 8)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [1, 0])
        )
        ones = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=([1, 1, 1], 1.0)
        )
        # Broadcast [8, 1] + [1, 1, 1] -> [1, 8, 1]: rank 3 under a rank 2 permute
        add = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(p1, ones))
        sl = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor, args=(add, 2, 0, 1)
        )
        # Sink view: input [1, 8, 1] has a single non-unit dim
        sink = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(sl, [8])
        )
        builder.output([sink])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        # Should not crash, and should skip the subgraph due to rank mismatch
        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 1
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "broadcast_rank_increase_no_crash",
        )


# ──────────────────────────────────────────────────────────────────────
# Tests for RemovePermutesAroundElementwiseOps
# ──────────────────────────────────────────────────────────────────────


class RemovePermutesAroundRepeatInterleaveTest(unittest.TestCase):
    """repeat_interleave lowers to unsqueeze -> expand_copy -> merging view_copy.

    The triple is rank-preserving, so a permutation flows through it once the
    interleaved dim is remapped. Shapes here mirror torchaudio's Stretch2d as it
    appears in wavernn between two channels-last convolutions.
    """

    @staticmethod
    def _interleave(
        builder: GraphBuilder,
        inp: object,
        shape: list[int],
        dim: int,
        scale: int,
    ) -> object:
        """Emit unsqueeze(dim+1) -> expand_copy(scale) -> view_copy(merge)."""
        unsqueezed = list(shape)
        unsqueezed.insert(dim + 1, 1)
        expanded = list(unsqueezed)
        expanded[dim + 1] = scale
        merged = list(shape)
        merged[dim] *= scale

        u = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(inp, dim + 1)
        )
        e = builder.call_operator(
            op=exir_ops.edge.aten.expand_copy.default, args=(u, expanded)
        )
        return builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(e, merged)
        )

    def test_removes_permutes_around_repeat_interleave(self) -> None:
        """permute(NHWC->NCHW) -> interleave(W) -> permute(NCHW->NHWC):
        both permutes should cancel and the interleave move to the NHWC dim."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 20, 1)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        v = self._interleave(builder, p1, [1, 1, 16, 20], dim=3, scale=2)
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.expand_copy.default), 1
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "RepeatInterleavePermutes"
        )

    def test_repeat_interleave_with_keyword_arguments(self) -> None:
        """The matcher and rewrite support schema arguments passed by keyword."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 20, 1)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        u = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default,
            args=(p1,),
            kwargs={"dim": 4},
        )
        e = builder.call_operator(
            op=exir_ops.edge.aten.expand_copy.default,
            args=(u,),
            kwargs={"size": [1, 1, 16, 20, 2]},
        )
        v = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(e,),
            kwargs={"size": [1, 1, 16, 40]},
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = cast(PassResult, RemovePermutesAroundElementwiseOps()(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "RepeatInterleaveKwargs"
        )

    def test_repeat_interleave_rank_mismatch_is_not_rewritten(self) -> None:
        """Reject a triple whose active boundary permutation has the wrong rank."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 20, 1)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        interleaved = self._interleave(builder, p1, [1, 1, 16, 20], dim=3, scale=2)
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(interleaved, [0, 2, 3, 1]),
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)
        start_permute = original.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        )[0]

        class RankMismatchPass(RemovePermutesAroundElementwiseOps):
            def get_permutation(self, node: torch.fx.Node) -> list[int] | None:
                if node is start_permute:
                    return [0, 2, 1]
                return super().get_permutation(node)

        result = cast(PassResult, RankMismatchPass()(original))
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )
        validate_numerics(
            gm_before,
            result.graph_module,
            [x_data],
            "RepeatInterleaveRankMismatch",
        )

    def test_repeat_interleave_after_nop_stretch(self) -> None:
        """The wavernn region verbatim: a freq_scale=1 Stretch2d leaves a nop
        unsqueeze/view pair ahead of the real interleave. The permutation must
        round-trip across it and still reach the triple."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 20, 1)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        u0 = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(p1, 3)
        )
        v0 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(u0, [1, 1, 16, 20])
        )
        v = self._interleave(builder, v0, [1, 1, 16, 20], dim=3, scale=2)
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = _canonicalize_and_remove_permutes(original)
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "RepeatInterleaveNopStretch"
        )

    def test_repeat_interleave_scale_four_on_middle_dim(self) -> None:
        """Interleaving a non-trailing dim with a scale other than 2."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 20, 3)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        v = self._interleave(builder, p1, [1, 3, 16, 20], dim=2, scale=4)
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "RepeatInterleaveMiddleDim"
        )

    def test_repeat_interleave_composes_with_elementwise(self) -> None:
        """An interleave and a pointwise op in the same permuted region."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 20, 1)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(p1, p1))
        v = self._interleave(builder, mul, [1, 1, 16, 20], dim=3, scale=2)
        relu = builder.call_operator(op=exir_ops.edge.aten.hardtanh.default, args=(v,))
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(relu, [0, 2, 3, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "RepeatInterleaveElementwise"
        )

    def test_resnet_stretch_region(self) -> None:
        """wavernn's resnet_stretch: the region enters through an unsqueeze at a
        position the permutation moves, and leaves through squeeze_copy.dims."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 8, 16)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 1])
        )
        u1 = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(p1, 1)
        )
        # freq_scale=1 Stretch2d leaves a nop unsqueeze/view pair.
        u2 = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(u1, 3)
        )
        v2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(u2, [1, 1, 16, 8])
        )
        v3 = self._interleave(builder, v2, [1, 1, 16, 8], dim=3, scale=4)
        sq = builder.call_operator(
            op=exir_ops.edge.aten.squeeze_copy.dims, args=(v3, [1])
        )
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(sq, [0, 2, 1])
        )
        builder.output([p2])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = _canonicalize_and_remove_permutes(original)
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
        validate_numerics(
            gm_before, result.graph_module, [x_data], "ResnetStretchRegion"
        )

    def test_non_merging_view_after_expand_is_not_optimized(self) -> None:
        """A view that is not the (dim, dim+1) merge is not layout-invariant,
        so the region must be left alone."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 20, 1)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        u = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(p1, 4)
        )
        e = builder.call_operator(
            op=exir_ops.edge.aten.expand_copy.default, args=(u, [1, 1, 16, 20, 2])
        )
        # Squeezes the leading dim instead of merging dims 3 and 4.
        v = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(e, [1, 16, 20, 2])
        )
        builder.output([v])
        original = builder.get_graph_module()

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 1
        )

    def test_expand_with_extra_user_is_not_optimized(self) -> None:
        """Rewriting the triple in place would corrupt a second consumer of the
        expand, so the triple must not be claimed."""
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 20, 1)
        x = builder.placeholder("x", x_data)
        p1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 3, 1, 2])
        )
        u = builder.call_operator(
            op=exir_ops.edge.aten.unsqueeze_copy.default, args=(p1, 4)
        )
        e = builder.call_operator(
            op=exir_ops.edge.aten.expand_copy.default, args=(u, [1, 1, 16, 20, 2])
        )
        v = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(e, [1, 1, 16, 40])
        )
        other = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(e, e))
        p2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(v, [0, 2, 3, 1])
        )
        builder.output([p2, other])
        original = builder.get_graph_module()

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 2
        )


class RemovePermutesAroundElementwiseOpsTest(unittest.TestCase):
    def test_no_permutes_is_noop(self) -> None:
        """With no surrounding permutes, the pass makes no change."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 4, 8, 8))
        mul = builder.call_operator(op=exir_ops.edge.aten.mul.Tensor, args=(x, x))
        builder.output([mul])
        original = builder.get_graph_module()

        p = RemovePermutesAroundElementwiseOps()
        result = cast(PassResult, p(original))
        self.assertFalse(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.permute_copy.default), 0
        )
