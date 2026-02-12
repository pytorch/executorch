# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import cast

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.fuse_ops import FuseQuantDequantToRequantizePass
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.pass_utils import (
    count_node,
    get_compute_nodes_in_gm,
    get_node_pos,
    nodes_not_adjacent_in_gm,
    nodes_not_connected_in_gm,
)
from executorch.backends.cadence.aot.reorder_ops import (
    AdvanceQuantizeOpAboveDefChainPass,
    AdvanceQuantizeOpAboveDefInBranchPass,
    PostponeDequantizeOpBelowUseChainPass,
    PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassBase, PassResult
from torch.utils import _pytree as pytree


def transform_and_check_numerics(
    original_graph: torch.fx.GraphModule,
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    pass_to_run: PassBase,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> PassResult:
    """Run a graph transformation and validate numerical equivalence.

    Args:
        original_graph: The original graph module before transformation
        inputs: Input tensors to run through both graphs
        pass_to_run: The pass to apply to the graph
        pass_name: Name of the pass being validated (for error messages)
        rtol: Relative tolerance for allclose comparison
        atol: Absolute tolerance for allclose comparison

    Returns:
        The PassResult from the transformation
    """
    # Deepcopy to preserve original for comparison
    gm_before = copy.deepcopy(original_graph)

    # Run the transformation
    result = cast(PassResult, pass_to_run.call(original_graph))

    # Validate numerical equivalence
    gm_before.eval()
    result.graph_module.eval()
    with torch.no_grad():
        orig_out = gm_before(*inputs)
        mod_out = result.graph_module(*inputs)

    flat_orig_out, _ = pytree.tree_flatten(orig_out)
    flat_mod_out, _ = pytree.tree_flatten(mod_out)

    # Check that outputs match within tolerance
    for i, (orig_tensor, mod_tensor) in enumerate(zip(flat_orig_out, flat_mod_out)):
        if not torch.allclose(orig_tensor, mod_tensor, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(orig_tensor - mod_tensor)).item()
            pass_name = type(pass_to_run).__name__
            raise AssertionError(
                f"Pass validation failed for pass {pass_name}. "
                f"Output tensor {i} differs by max {max_diff:.6e}. "
                f"Expected rtol={rtol}, atol={atol}. "
                f"Original output: {orig_tensor}, Modified output: {mod_tensor}"
            )

    return result


class TestReorderPasses(unittest.TestCase):
    def test_advance_branched_quantize(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(64, 3, dtype=torch.float32))
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(x, [32, 6]),
        )
        aten_slice_copy_tensor = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(view, 0, 0, 6),
        )
        quantized_decomposed_quantize_per_tensor_default = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(aten_slice_copy_tensor, 0.1, 10, 0, 255, torch.uint8),
        )

        aten_slice_copy_tensor_1 = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(view, 0, 6, 12),
        )
        quantized_decomposed_quantize_per_tensor_default_1 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(aten_slice_copy_tensor_1, 0.1, 10, 0, 255, torch.uint8),
        )

        aten_slice_copy_tensor_2 = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(view, 0, 12, 18),
        )
        quantized_decomposed_quantize_per_tensor_default_2 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(aten_slice_copy_tensor_2, 0.1, 10, 0, 255, torch.uint8),
        )

        aten_slice_copy_tensor_3 = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(view, 0, 18, 24),
        )
        quantized_decomposed_quantize_per_tensor_default_3 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(aten_slice_copy_tensor_3, 0.2, 4, 0, 255, torch.uint8),
        )
        builder.output(
            [
                quantized_decomposed_quantize_per_tensor_default,
                quantized_decomposed_quantize_per_tensor_default_1,
                quantized_decomposed_quantize_per_tensor_default_2,
                quantized_decomposed_quantize_per_tensor_default_3,
            ]
        )
        original_graph = builder.get_graph_module()
        p = AdvanceQuantizeOpAboveDefInBranchPass()
        graph_module = cast(PassResult, p(original_graph)).graph_module
        graph_module.graph.eliminate_dead_code()
        nodes = get_compute_nodes_in_gm(graph_module)
        # The quantize op should be hoisted to dominate the branch
        self.assertTrue(
            nodes[0] == exir_ops.edge.quantized_decomposed.quantize_per_tensor
        )
        # There should be 5 quantize ops: the 4 originally present in the model,
        # and the one that was hoisted above the slices
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            ),
            5,
        )
        # Ensure none of the slice nodes were erroneously removed
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.aten.slice_copy.Tensor,
            ),
            4,
        )
        # Each of the 4 original quant ops should now be paired with a dequant op
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
            4,
        )
        p = FuseQuantDequantToRequantizePass()
        graph_module = cast(PassResult, p(graph_module)).graph_module
        # We expect 3 dequant/quant pairs to be removed because they have matching params,
        # leaving a single dequant/quant pair that is then merged into a requantize op
        self.assertEqual(
            count_node(
                graph_module,
                exir_ops.edge.cadence.requantize.per_tensor,
            ),
            1,
        )

    @torch.no_grad()
    def test_advance_quantize(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(16, 1, 32, 6, dtype=torch.float32)
        weight_data = torch.randint(-128, 127, (32, 32), dtype=torch.int8)
        x = builder.placeholder("x", x_data)
        weights = builder.placeholder("weights", weight_data)
        full = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], -7),
            kwargs={"dtype": torch.int32},
        )
        full_1 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 1525501056),
        )
        full_2 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 2),
        )
        full_3 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 0),
            kwargs={"dtype": torch.int32},
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [1, 0, 3, 2]),
        )
        quantize_per_tensor = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(permute, 0.029049983248114586, -1, -128, 127, torch.int8),
        )
        quantized_linear = builder.call_operator(
            op=exir_ops.edge.cadence.quantized_linear.default,
            args=(
                quantize_per_tensor,
                weights,
                full_3,
                -1,
                full_2,
                full_1,
                full,
                -7,
                None,
            ),
        )
        dequantize_per_tensor = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(quantized_linear, 0.01627226173877716, -7, -128, 127, torch.int8),
        )
        builder.output([dequantize_per_tensor])
        original_graph = builder.get_graph_module()

        p1 = AdvanceQuantizeOpAboveDefInBranchPass()
        tmp_graph = cast(PassResult, p1(original_graph)).graph_module
        result = transform_and_check_numerics(
            tmp_graph,
            (x_data, weight_data),
            AdvanceQuantizeOpAboveDefChainPass(),
        )
        self.assertFalse(result.modified)
        converted_graph = result.graph_module
        # Assert that permute node is now the successor of the quant node.
        self.assertTrue(
            get_node_pos(
                converted_graph, exir_ops.edge.cadence.quantize_per_tensor.default
            )
            < get_node_pos(converted_graph, exir_ops.edge.aten.permute_copy.default)
        )

    def test_postpone_dequantize1(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(1, 16, 32, 6, dtype=torch.float32)
        weight_data = torch.randint(-128, 127, (6, 6), dtype=torch.int8)
        x = builder.placeholder("x", x_data)
        weights = builder.placeholder("weights", weight_data)
        full = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], -7),
            kwargs={"dtype": torch.int32},
        )
        full_1 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 1461148032),
        )
        full_2 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], -4),
        )
        full_3 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 0),
            kwargs={"dtype": torch.int32},
        )
        quantize_per_tensor = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 0.029049983248114586, -1, -128, 127, torch.int8),
        )
        quantized_linear = builder.call_operator(
            op=exir_ops.edge.cadence.quantized_linear.default,
            args=(
                quantize_per_tensor,
                weights,
                full_3,
                -8,
                full_2,
                full_1,
                full,
                0,
                None,
            ),
        )
        dequantize_per_tensor = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(quantized_linear, 0.01627226173877716, -7, -128, 127, torch.int8),
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(dequantize_per_tensor, [1, 0, 3, 2]),
        )
        builder.output([permute])
        original_graph = builder.get_graph_module()
        result = transform_and_check_numerics(
            original_graph,
            (x_data, weight_data),
            PostponeDequantizeOpBelowUseChainPass(),
        )
        self.assertTrue(result.modified)
        converted_graph = result.graph_module
        # Assert that dequant node is now the successor of the permute node.
        self.assertTrue(
            get_node_pos(converted_graph, exir_ops.edge.aten.permute_copy.default)
            < get_node_pos(
                converted_graph,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            )
        )

    def test_postpone_dequantize_branched(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randint(0, 255, [1, 18, 3], dtype=torch.uint8)
        weights_data = torch.randn([3, 3], dtype=torch.float32)

        x = builder.placeholder("x", x_data)
        p_linear_weight = builder.placeholder("weights", weights_data)
        quantized_decomposed_dequantize_per_tensor_default = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(x, 0.1, 10, 0, 255, torch.uint8),
        )
        aten_squeeze_copy_dims = builder.call_operator(
            op=exir_ops.edge.aten.squeeze_copy.dims,
            args=(quantized_decomposed_dequantize_per_tensor_default, [0]),
        )

        aten_slice_copy_tensor = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(aten_squeeze_copy_dims, 0, 0, 6),
        )
        aten_permute_copy_default = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(p_linear_weight, [1, 0]),
        )
        aten_mm_default = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(aten_slice_copy_tensor, aten_permute_copy_default),
        )

        aten_slice_copy_tensor_1 = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(aten_squeeze_copy_dims, 0, 6, 12),
        )
        aten_permute_copy_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(p_linear_weight, [1, 0]),
        )
        aten_mm_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(aten_slice_copy_tensor_1, aten_permute_copy_default_1),
        )

        aten_slice_copy_tensor_2 = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(aten_squeeze_copy_dims, 0, 12, 18),
        )
        aten_permute_copy_default_2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(p_linear_weight, [1, 0]),
        )
        aten_mm_default_2 = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(aten_slice_copy_tensor_2, aten_permute_copy_default_2),
        )
        builder.output([aten_mm_default, aten_mm_default_1, aten_mm_default_2])
        original_graph = builder.get_graph_module()
        result = transform_and_check_numerics(
            original_graph,
            (x_data, weights_data),
            PostponeDequantizeOpBelowUseChainPass(),
        )
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Asset that the dequant node was split into 4, one per branch
        self.assertEqual(
            count_node(
                converted_graph,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ),
            3,
        )

        # Assert that the dequant node is no longer the predecessor of the squeeze node
        self.assertTrue(
            nodes_not_connected_in_gm(
                converted_graph,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                exir_ops.edge.aten.squeeze_copy.dims,
            ),
        )
        # Assert that dequant node is not predecessor of slice (it should've been moved below slice)
        self.assertTrue(
            nodes_not_connected_in_gm(
                converted_graph,
                exir_ops.edge.cadence.dequantize_per_tensor.default,
                exir_ops.edge.aten.slice_copy.Tensor,
            ),
        )

    # 4d -> permute -> 4d -> view -> 3d
    def test_permute3_view4_chains(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(3, 1, 768)
        x = builder.placeholder("x", x_data)
        aten_view_copy_default = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(x, [3, 12, 64]),
        )
        aten_permute_copy_default = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(aten_view_copy_default, [1, 0, 2]),
        )
        aten_view_copy_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(aten_permute_copy_default, [1, 12, 3, 64]),
        )
        aten_permute_copy_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(aten_view_copy_default_1, [0, 1, 3, 2]),
        )
        builder.output([aten_permute_copy_default_1])
        original_graph = builder.get_graph_module()
        result = transform_and_check_numerics(
            original_graph,
            (x_data,),
            PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(),
        )
        self.assertTrue(result.modified)
        converted_graph = result.graph_module
        converted_graph.graph.eliminate_dead_code()
        # Assert the order becomes view, view, permute, permute
        nodes = get_compute_nodes_in_gm(converted_graph)
        self.assertEqual(len(nodes), 4)
        self.assertTrue(nodes[0] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[1] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[2] == exir_ops.edge.aten.permute_copy)
        self.assertTrue(nodes[3] == exir_ops.edge.aten.permute_copy)

    # 3d -> permute -> 3d -> view -> 4d
    def test_permute4_view3_chains(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(3, 1, 768)
        x = builder.placeholder("x", x_data)
        aten_view_copy_default = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(x, [1, 3, 12, 64]),
        )
        aten_permute_copy_default = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(aten_view_copy_default, [3, 1, 0, 2]),
        )
        aten_view_copy_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(aten_permute_copy_default, [64, 3, 12]),
        )
        aten_permute_copy_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(aten_view_copy_default_1, [2, 1, 0]),
        )
        builder.output([aten_permute_copy_default_1])
        original_graph = builder.get_graph_module()

        result = transform_and_check_numerics(
            original_graph,
            (x_data,),
            PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(),
        )
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Assert the order becomes view, view, permute, permute
        nodes = get_compute_nodes_in_gm(converted_graph)
        self.assertEqual(len(nodes), 4)
        self.assertTrue(nodes[0] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[1] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[2] == exir_ops.edge.aten.permute_copy)
        self.assertTrue(nodes[3] == exir_ops.edge.aten.permute_copy)

    # Negative test case where the transform should not happen.
    # permute->4d->view->3d where the view not only removes the dimension whose
    # size is 1 (this is ok), but also changes the size of the dimensions (not ok).
    def test_permute_view_chains_neg(self) -> None:
        builder = GraphBuilder()
        x_data = torch.randn(3, 1, 768)
        x = builder.placeholder("x", x_data)
        aten_view_copy_default = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(x, [1, 3, 12, 64]),
        )
        aten_permute_copy_default = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(aten_view_copy_default, [3, 1, 0, 2]),
        )
        aten_view_copy_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(aten_permute_copy_default, [64, 6, 6]),
        )
        aten_permute_copy_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(aten_view_copy_default_1, [2, 1, 0]),
        )
        builder.output([aten_permute_copy_default_1])
        original_graph = builder.get_graph_module()

        # Performing transform (nothing should happen)
        result = transform_and_check_numerics(
            original_graph,
            (x_data,),
            PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(),
        )
        self.assertFalse(result.modified)
        converted_graph = result.graph_module

        # Assert the order is still view, permute, view, permute
        nodes = get_compute_nodes_in_gm(converted_graph)
        self.assertEqual(len(nodes), 4)
        self.assertTrue(nodes[0] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[1] == exir_ops.edge.aten.permute_copy)
        self.assertTrue(nodes[2] == exir_ops.edge.aten.view_copy)
        self.assertTrue(nodes[3] == exir_ops.edge.aten.permute_copy)
