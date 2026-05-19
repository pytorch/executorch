# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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
from executorch.backends.transforms.postpone_permute_below_squeeze_view import (
    PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView,
)
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.backends.transforms.replace_nop_transpose_or_permute_with_view import (
    ReplaceNopTransposeOrPermuteWithViewPass,
)
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
        # permute2 reverses permute1 => identity
        permute2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(permute1, [0, 3, 1, 2])
        )
        # permute3: different permutation
        permute3 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(permute1, [0, 2, 1, 3])
        )
        # permute4 -> permute5: chained
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
        # Verify order: views before permutes
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
        # Order unchanged: view, permute, view, permute
        targets = get_compute_nodes(result.graph_module)
        self.assertEqual(targets[0], exir_ops.edge.aten.view_copy.default)
        self.assertEqual(targets[1], exir_ops.edge.aten.permute_copy.default)


# ──────────────────────────────────────────────────────────────────────
# Tests for ReplaceNopTransposeOrPermuteWithViewPass
# ──────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────
# Tests for RemovePermutesAroundElementwiseOps cross-view handling
# ──────────────────────────────────────────────────────────────────────


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
        Tests the explicit unsqueeze_copy/squeeze_copy code paths in
        _adapt_permute_across_view (distinct from view_copy)."""
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
            "permute_unsqueeze_copy_mul_squeeze_copy_permute",
        )

    def test_4d_permute_squeeze_copy_clamp_3d_permute(self) -> None:
        """4D permute([0,3,1,2]) → squeeze_copy(dim=2) → hardtanh → 3D permute([0,2,1]).
        Tests the squeeze_copy code path at the start boundary (entering the
        subgraph via squeeze_copy rather than view_copy)."""
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
            "permute_unsqueeze_copy_neg_dim_mul_squeeze_copy_permute",
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
