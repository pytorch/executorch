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
