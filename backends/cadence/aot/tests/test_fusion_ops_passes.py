# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import copy
import unittest
from typing import cast, Final, List, Tuple

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.fuse_ops import (
    FuseCascadedTransposeOrPermuteOps,
    FuseCascadedViewOps,
    FuseFullThenReshapePass,
    FuseMMWithAdd,
    FuseMulScalarIntoDequantPass,
    FuseMulTensorIntoDequantPass,
    FuseMulTensorIntoQuantPass,
    FuseQuantDequantToRequantizePass,
    FuseTransposeOrPermuteOpPairsPass,
)
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.pass_utils import count_node, op_counts_match
from executorch.backends.cadence.aot.typing_stubs import expand
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult, ProxyValue
from torch.utils import _pytree as pytree


def validate_numerics(
    original: torch.fx.GraphModule,
    modified: torch.fx.GraphModule,
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    pass_name: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Validate that two graph modules produce numerically equivalent outputs.

    Args:
        original: The original graph module before the pass
        modified: The modified graph module after the pass
        inputs: Input tensors to run through both graphs
        pass_name: Name of the pass being validated (for error messages)
        rtol: Relative tolerance for allclose comparison
        atol: Absolute tolerance for allclose comparison
    """
    original.eval()
    modified.eval()
    with torch.no_grad():
        orig_out = original(*inputs)
        mod_out = modified(*inputs)

    flat_orig_out, _ = pytree.tree_flatten(orig_out)
    flat_mod_out, _ = pytree.tree_flatten(mod_out)

    # Check that outputs match within tolerance
    for i, (orig_tensor, mod_tensor) in enumerate(zip(flat_orig_out, flat_mod_out)):
        if not torch.allclose(orig_tensor, mod_tensor, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(orig_tensor - mod_tensor)).item()
            raise AssertionError(
                f"Pass validation failed for pass {pass_name}. "
                f"Output tensor {i} differs by max {max_diff:.6e}. "
                f"Expected rtol={rtol}, atol={atol}. "
                f"Original output: {orig_tensor}, Modified output: {mod_tensor}"
            )


class TestFusionPassesBase(unittest.TestCase):
    def check_op_counts(
        self,
        graph_module: torch.fx.GraphModule,
        expected_op_counts: dict[EdgeOpOverload, int],
    ) -> None:
        self.assertTrue(op_counts_match(graph_module, expected_op_counts))


class TestFuseMMWithAddPass(TestFusionPassesBase):
    def test_no_fuse_for_3d_bias(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(4, 3, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(3, 5, dtype=torch.float32))
        z = builder.placeholder("z", torch.randn(1, 4, 5, dtype=torch.float32))
        mm = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(x, y),
        )
        output = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(mm, z))
        builder.output([output])
        original_graph = builder.get_graph_module()

        p = FuseMMWithAdd()
        result = cast(PassResult, p(original_graph))
        self.assertFalse(result.modified)
        converted_graph = result.graph_module
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.addmm.default), 0
        )
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.mm.default), 1)
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.add.Tensor), 1)

    def test_fuse_mm_with_add(self) -> None:
        builder = GraphBuilder()
        x_input = torch.randn(3, 5, dtype=torch.float32)
        y_input = torch.randn(5, 6, dtype=torch.float32)
        z_input = torch.randn(6, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        y = builder.placeholder("y", y_input)
        z = builder.placeholder("z", z_input)
        mm = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(x, y),
        )
        output = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(mm, z))
        builder.output([output])
        original_graph = builder.get_graph_module()
        gm_before = copy.deepcopy(original_graph)

        p = FuseMMWithAdd()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Validate numerical accuracy
        validate_numerics(
            gm_before, converted_graph, (x_input, y_input, z_input), "FuseMMWithAdd"
        )

        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.addmm.default), 1
        )
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.mm.default), 0)
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.add.Tensor), 0)

    def test_fuse_view_mm_view_add(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(4, 8, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(2, 4, 6, dtype=torch.float32))
        z = builder.placeholder("z", torch.randn(6, dtype=torch.float32))
        y_view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(y, [8, 6])
        )
        mm = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(x, y_view),
        )
        mm_view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mm, [2, 2, 6])
        )
        output = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(mm_view, z)
        )
        builder.output([output])
        original_graph = builder.get_graph_module()

        p = FuseMMWithAdd()
        converted_graph = cast(PassResult, p(original_graph)).graph_module
        converted_graph.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.addmm.default), 1
        )
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.mm.default), 0)
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.add.Tensor), 0)

    def test_keep_view_mm_view_add(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(4, 8, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(2, 4, 6, dtype=torch.float32))
        # Bias is not broadcastable to output of mm
        z = builder.placeholder("z", torch.randn(2, 2, 1, dtype=torch.float32))
        y_view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(y, [8, 6])
        )
        mm = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(x, y_view),
        )
        mm_view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mm, [2, 2, 6])
        )
        output = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(mm_view, z)
        )
        builder.output([output])
        original_graph = builder.get_graph_module()
        p = FuseMMWithAdd()
        converted_graph = cast(PassResult, p(original_graph)).graph_module
        converted_graph.graph.eliminate_dead_code()
        # Assert that mm and add were not fused to addmm, since z cannot be
        # broadcasted to the out of mm.
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.addmm.default), 0
        )
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.mm.default), 1)
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.add.Tensor), 1)

    def test_fuse_mm_add_with_bias(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(3, 5, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(5, 6, dtype=torch.float32))
        mm = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(x, y),
        )
        bias = builder.call_operator(op=exir_ops.edge.aten.full.default, args=([1], 1))
        output = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(mm, bias)
        )
        builder.output([output])
        original_graph = builder.get_graph_module()
        p = FuseMMWithAdd()
        converted_graph = cast(PassResult, p(original_graph)).graph_module
        converted_graph.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.addmm.default), 1
        )
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.mm.default), 0)
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.add.Tensor), 0)

    def test_keep_mm_add_with_multiple_users(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(3, 5, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(5, 6, dtype=torch.float32))
        z = builder.placeholder("z", torch.randn(6, dtype=torch.float32))
        mm = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(x, y),
        )
        # The add consuming the output of mm has more than one users.
        add1 = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(mm, z))
        add2 = builder.call_operator(op=exir_ops.edge.aten.add.Tensor, args=(add1, z))
        output = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor, args=(add1, add2)
        )
        builder.output([output])
        original_graph = builder.get_graph_module()
        p = FuseMMWithAdd()
        converted_graph = cast(PassResult, p(original_graph)).graph_module
        converted_graph.graph.eliminate_dead_code()
        # Assert that mm and add were not fused to addmm, since add has multiple
        # users.
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.addmm.default), 0
        )
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.mm.default), 1)
        self.assertEqual(count_node(converted_graph, exir_ops.edge.aten.add.Tensor), 3)


class TestFusionPasses(TestFusionPassesBase):
    def test_permute_transpose_fusion(self) -> None:
        builder = GraphBuilder()
        x_input = torch.randn(3, 1, 3, 1, 4, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(x, [0, 2, 4, 1, 3])
        )
        output = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(permute, 1, 0),
        )
        builder.output([output])
        original_graph = builder.get_graph_module()
        graph_copy = copy.deepcopy(original_graph)
        p = FuseCascadedTransposeOrPermuteOps()
        result = p.call(original_graph)
        self.assertTrue(result.modified)
        converted_graph = result.graph_module
        converted_graph.graph.eliminate_dead_code()
        # Assert that permute op was fused with transpose op
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.permute_copy.default), 1
        )
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.transpose_copy.int), 0
        )
        validate_numerics(
            graph_copy, converted_graph, (x_input,), "FuseCascadedTransposeOrPermuteOps"
        )

    def test_view_fusion(self) -> None:
        builder = GraphBuilder()
        x_input = torch.randn(8, 5, 3, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        view1 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(x, [1, 8, 15])
        )
        view2 = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(view1, [1, 1, 120])
        )
        output = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(view2, [1, 12, 10])
        )
        builder.output([output])
        original_graph = builder.get_graph_module()

        gm_before = copy.deepcopy(original_graph)
        p = FuseCascadedViewOps()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Validate numerical accuracy
        inputs = [x_input]
        validate_numerics(gm_before, converted_graph, inputs, "FuseCascadedViewOps")

        # Assert that only one view op remains
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.view_copy.default), 1
        )

    def test_view_fusion_branched(self) -> None:
        builder = GraphBuilder()
        x_input = torch.randn(8, 5, 3, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        y = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(x, [1, 8, 15])
        )
        z = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(y, [1, 1, 120])
        )
        t = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(y, [120, 1, 1])
        )
        builder.output([z, t])
        original_graph = builder.get_graph_module()

        gm_before = copy.deepcopy(original_graph)
        p = FuseCascadedViewOps()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Validate numerical accuracy
        inputs = [x_input]
        validate_numerics(gm_before, converted_graph, inputs, "FuseCascadedViewOps")

        # z and t should be fused and y should be eliminated.
        self.assertEqual(
            count_node(converted_graph, exir_ops.edge.aten.view_copy.default), 2
        )

    def test_force_quant_dequant_fusion(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 12, 1, 6, dtype=torch.float32))
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 1.2, 3, 0, 127, torch.int8),
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(quant, [2, 0, 1, 3])
        )
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(permute, 4.5, 6, 0, 127, torch.int8),
        )
        builder.output([dequant])
        original_graph = builder.get_graph_module()
        p = FuseQuantDequantToRequantizePass(force_quant_dequant_fusion=True)
        converted_graph = cast(PassResult, p(original_graph)).graph_module
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                # Verify that dequant/quant pair was replaced with requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 0,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                exir_ops.edge.cadence.requantize.per_tensor: 1,
            },
        )

    def test_no_replace_quant_permute_dequant_with_requantize(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 12, 1, 6, dtype=torch.float32))
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 1.2, 3, 0, 127, torch.int8),
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(quant, [2, 0, 1, 3])
        )
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(permute, 4.5, 6, 0, 127, torch.int8),
        )
        builder.output([dequant])
        original_graph = builder.get_graph_module()

        p = FuseQuantDequantToRequantizePass(force_quant_dequant_fusion=False)
        result = cast(PassResult, p(original_graph))
        self.assertFalse(result.modified)
        converted_graph = result.graph_module
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                # Verify that no dequant/quant pair was replaced with requantize.
                # quantize -> permute -> dequantize should not be replaced with requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                exir_ops.edge.cadence.requantize.per_tensor: 0,
            },
        )

    def test_replace_quant_view_dequant_with_requantize(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 12, 1, 6, dtype=torch.float32))
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 1.2, 3, 0, 127, torch.int8),
        )
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(quant, [-1])
        )
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(view, 4.5, 6, 0, 127, torch.int8),
        )
        builder.output([dequant])
        original_graph = builder.get_graph_module()
        p = FuseQuantDequantToRequantizePass()
        converted_graph = cast(PassResult, p(original_graph)).graph_module
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                # Verify that dequant/quant pair was replaced with requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 0,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                exir_ops.edge.cadence.requantize.per_tensor: 1,
            },
        )

    def test_replace_dequant_quant_with_requantize(self) -> None:
        builder = GraphBuilder()
        x_input = torch.randint(low=0, high=5, size=(2, 12, 1, 6), dtype=torch.int8)
        x = builder.placeholder("x", x_input)
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(x, 1.2, 3, 0, 127, torch.int8),
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant, 4.5, 6, 0, 127, torch.int8),
        )
        builder.output([quant])
        original_graph = builder.get_graph_module()
        gm_before = copy.deepcopy(original_graph)

        p = FuseQuantDequantToRequantizePass()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Validate numerical accuracy
        validate_numerics(
            gm_before, converted_graph, (x_input,), "FuseQuantDequantToRequantizePass"
        )

        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                # Verify that dequant -> quant was replaced with requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 0,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                exir_ops.edge.cadence.requantize.per_tensor: 1,
            },
        )

    def test_replace_dequant_permute_quant_with_requantize(self) -> None:
        builder = GraphBuilder()
        x_input = torch.randint(low=0, high=5, size=(2, 12, 1, 6), dtype=torch.int8)
        x = builder.placeholder("x", x_input)
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(x, 1.2, 3, 0, 127, torch.int8),
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(dequant, [2, 0, 1, 3])
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(permute, 4.5, 6, 0, 127, torch.int8),
        )
        builder.output([quant])
        original_graph = builder.get_graph_module()
        gm_before = copy.deepcopy(original_graph)

        p = FuseQuantDequantToRequantizePass()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Validate numerical accuracy
        validate_numerics(
            gm_before, converted_graph, (x_input,), "FuseQuantDequantToRequantizePass"
        )

        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                # Verify that dequant -> permute -> quant was replaced with permute -> requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 0,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                exir_ops.edge.aten.permute_copy.default: 1,
                exir_ops.edge.cadence.requantize.per_tensor: 1,
            },
        )

    def test_remove_nop_dequant_quant(self) -> None:
        leading_dims = 12
        in_dim = 6
        out_dim = 12

        builder = GraphBuilder()
        x = builder.placeholder(
            "x", torch.randn(leading_dims, in_dim, dtype=torch.float32)
        )
        quant1 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 4.5, 6, 0, 127, torch.int8),
        )
        weights = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=([out_dim, in_dim], 1)
        )
        bias = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=([out_dim], 1)
        )
        weight_zero_point = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=([in_dim], 0)
        )
        out_multiplier = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=([out_dim], 1)
        )
        out_shift = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=([out_dim], 0)
        )
        linear1 = builder.call_operator(
            op=exir_ops.edge.cadence.quantized_linear.default,
            args=(
                quant1,
                weights,
                bias,
                0,  # src_zero_point
                weight_zero_point,
                out_multiplier,
                out_shift,
                0,  # out_zero_point
                None,
            ),
        )
        dequant1 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(linear1, 1.2, 3, 0, 127, torch.int8),
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default, args=(dequant1, [1, 0])
        )
        quant2 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(permute, 4.5, 6, 0, 127, torch.int8),
        )
        linear2 = builder.call_operator(
            op=exir_ops.edge.cadence.quantized_linear.default,
            args=(
                quant2,
                weights,
                bias,
                0,  # src_zero_point
                weight_zero_point,
                out_multiplier,
                out_shift,
                0,  # out_zero_point
                None,
            ),
        )
        dequant2 = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(linear2, 1.2, 3, 0, 127, torch.int8),
        )
        builder.output([dequant2])
        original_graph = builder.get_graph_module()
        p = FuseQuantDequantToRequantizePass()
        converted_graph = cast(PassResult, p(original_graph)).graph_module
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                # Verify that one dequant/quant pair was removed from chain:
                # quant->linear->dequant->permute->quant->linear->dequant
                # gets converted to:
                # quant->linear->permute->linear->dequant
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
            },
        )

    def test_fuse_mul_into_dequant(self) -> None:
        INPUT_SHAPE: Final[List[int]] = [4, 32]
        DEQUANT_SCALE: Final[float] = 1.5
        FULL_VALUE: Final[float] = 3

        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*INPUT_SHAPE, dtype=torch.float32))
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(x, DEQUANT_SCALE, 0, 0, 255, torch.uint8),
        )
        full = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(INPUT_SHAPE, FULL_VALUE),
        )
        mul = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor,
            args=(dequant, full),
        )
        builder.output([mul])
        original_graph = builder.get_graph_module()
        p = FuseMulTensorIntoDequantPass()
        converted_graph = cast(PassResult, p(original_graph)).graph_module

        # verify that the mul and full ops were removed
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                exir_ops.edge.aten.full.default: 0,
                exir_ops.edge.aten.mul.Tensor: 0,
            },
        )

        # verify that the dequant scale value was updated correctly
        deq_scale = -1
        for node in converted_graph.graph.nodes:
            if (
                node.target
                == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ):
                deq_scale = node.args[1]
        self.assertEqual(deq_scale, DEQUANT_SCALE * FULL_VALUE)

    def test_fuse_mul_scalar_into_dequant(self) -> None:
        dequant_scale = 0.006
        mul_value = 0.3

        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 3, 4, dtype=torch.float32))
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 1, 0, -128, 127, torch.int8),
        )
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(quant, dequant_scale, 5, -128, 127, torch.int8),
        )
        mul_scalar = builder.call_operator(
            op=exir_ops.edge.aten.mul.Scalar,
            args=(dequant, mul_value),
        )
        builder.output([mul_scalar])
        original_graph = builder.get_graph_module()
        p = FuseMulScalarIntoDequantPass()
        converted_graph = cast(PassResult, p(original_graph)).graph_module

        # verify that the mul and full ops were removed
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                exir_ops.edge.aten.mul.Scalar: 0,
            },
        )

        # verify that the dequant scale value was updated correctly
        deq_scale = -1
        for node in converted_graph.graph.nodes:
            if (
                node.target
                == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ):
                deq_scale = node.args[1]
        self.assertEqual(deq_scale, dequant_scale * mul_value)

    def test_fuse_mul_into_quant(self) -> None:
        quant_scale = 5
        mul_value = 10

        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(4, 32, dtype=torch.float32))
        full = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], mul_value),
        )
        mul = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor,
            args=(x, full),
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(mul, quant_scale, 7, 0, 255, torch.uint8),
        )
        builder.output([quant])
        original_graph = builder.get_graph_module()
        p = FuseMulTensorIntoQuantPass()
        converted_graph = cast(PassResult, p(original_graph)).graph_module

        # verify that the mul and full ops were removed
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
                exir_ops.edge.aten.full.default: 0,
                exir_ops.edge.aten.mul.Tensor: 0,
            },
        )

        # verify that the quant scale value was updated correctly
        for node in converted_graph.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        ):
            new_quant_scale = node.args[1]
            self.assertEqual(new_quant_scale, quant_scale / mul_value)

        # verify the math is correct
        inp = torch.randn(4, 32, dtype=torch.float32)
        original_out = original_graph(inp)[0]
        new_out = converted_graph(inp)[0]
        assert torch.equal(original_out, new_out)

    def test_fuse_then_transpose_pass(self) -> None:
        # Create a graph with full -> transpose -> permute -> view.
        builder = GraphBuilder()
        full_node = builder.call_operator(
            op=exir_ops.edge.aten.full.default, args=((2, 3), 1)
        )
        transpose_node = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(full_node, 0, 1),
        )
        permute_node = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(transpose_node, (1, 0)),
        )
        view_node = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(permute_node, (1, 6, 1)),
        )
        builder.output([view_node])
        original_graph = builder.get_graph_module()
        gm_before = copy.deepcopy(original_graph)

        self.check_op_counts(
            original_graph,
            expected_op_counts={
                exir_ops.edge.aten.full.default: 1,
                exir_ops.edge.aten.transpose_copy.int: 1,
                exir_ops.edge.aten.permute_copy.default: 1,
                exir_ops.edge.aten.view_copy.default: 1,
            },
        )

        # Check that the pass fuses the full with all other ops (transpose, permute, view).
        p = FuseFullThenReshapePass()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        gm_after_pass = result.graph_module

        # Validate numerical accuracy
        validate_numerics(gm_before, gm_after_pass, [], "FuseFullThenReshapePass")

        self.check_op_counts(
            gm_after_pass,
            expected_op_counts={
                exir_ops.edge.aten.full.default: 1,
                exir_ops.edge.aten.transpose_copy.int: 0,
                exir_ops.edge.aten.permute_copy.default: 0,
                exir_ops.edge.aten.view_copy.default: 0,
            },
        )


class TestFuseTransposeOrPermuteOpPairsPass(TestFusionPassesBase):
    def _create_operator(
        self, builder: GraphBuilder, op: torch._ops.OpOverload, x: ProxyValue
    ) -> ProxyValue:
        if op == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default:
            return builder.call_operator(
                op=op,
                args=(x, 1.2, 3, 0, 127, torch.int8),
            )
        elif op == exir_ops.edge.cadence.quantized_relu.per_tensor:
            return builder.call_operator(
                op=op,
                args=(x, 0, 0, 0, 0),
            )
        else:
            raise ValueError(f"Unsupported op: {op}")


class TestFuseTransposeOpPairsPass(TestFusionPassesBase):
    def _create_operator(
        self, builder: GraphBuilder, op: torch._ops.OpOverload, x: ProxyValue
    ) -> ProxyValue:
        if op == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default:
            return builder.call_operator(
                op=op,
                args=(x, 1.2, 3, 0, 127, torch.int8),
            )
        elif op == exir_ops.edge.cadence.quantized_relu.per_tensor:
            return builder.call_operator(
                op=op,
                args=(x, 0, 0, 0, 0),
            )
        else:
            raise ValueError(f"Unsupported op: {op}")

    @expand(
        [
            # transpose -> quant -> same transpose => fuse
            (
                True,
                [0, 1],
                True,
                [0, 1],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                True,
            ),
            # same with different input size
            (
                True,
                [0, 1],
                True,
                [0, 1],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                True,
                [4, 4, 4],
            ),
            # transpose -> quant -> same transpose => fuse (same with transpose dimensions in different order, and with different skip quant op)
            (
                True,
                [0, 1],
                True,
                [1, 0],
                exir_ops.edge.cadence.quantized_relu.per_tensor,
                True,
            ),
            # transpose -> quant -> different transpose => don't fuse
            (
                True,
                [0, 1],
                True,
                [0, 2],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                False,
            ),
            # transpose -> quant -> transpose is not the reverse BUT there is a UNITARY dimension
            # so it ends up being the same on memory => fuse
            (
                True,
                [0, 1],
                True,
                [0, 2],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                True,
                [5, 40, 1],
            ),
            # transpose -> quant -> transpose is not the reverse, and unitary dimensions
            # don't help => don't fuse
            (
                True,
                [0, 1],
                True,
                [1, 3],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                False,
                [5, 40, 1, 4],
            ),
            # permutation -> quant -> opposite permutation => fuse
            (
                False,
                [1, 2, 0],
                False,
                [2, 0, 1],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                True,
            ),
            # same with different input size
            (
                False,
                [1, 2, 0],
                False,
                [2, 0, 1],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                True,
                [4, 4, 4],
            ),
            # permutation -> quant -> not the opposite permutation => don't fuse
            (
                False,
                [1, 2, 0],
                False,
                [1, 2, 0],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                False,
            ),
            # same with different input size
            (
                False,
                [1, 2, 0],
                False,
                [1, 2, 0],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                False,
                [4, 4, 4],
            ),
            # permutation -> quant -> a non reverse permutation BUT there is a UNITARY dimension
            # so it ends up being the same on memory => fuse
            (
                False,
                [1, 3, 2, 0],
                False,
                [3, 2, 1, 0],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                True,
                [3, 1, 8, 10],
            ),
            # permutation -> quant -> a non reverse permutation, and unitary dimensions
            # don't help => don't fuse
            (
                False,
                [1, 3, 2, 0],
                False,
                [3, 1, 2, 0],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                False,
                [3, 1, 8, 10],
            ),
            # transpose -> quant -> transpose as a permutation => fuse
            (
                True,
                [0, 1],
                False,
                [1, 0, 2],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                True,
            ),
            # transpose -> quant -> not opposite permutation => fuse
            (
                True,
                [0, 1],
                False,
                [0, 2, 1],
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                False,
            ),
        ],
    )
    def test_fuse_transpose_permute_pairs(
        self,
        is_op1_transpose: bool,
        perm1: list[int],
        is_op2_transpose: bool,
        perm2: list[int],
        quant_op: torch._ops.OpOverload,
        expected_is_fused: bool,
        dims: Tuple[int, int, int] = (2, 3, 4),
    ) -> None:
        # Create a graph with transpose/permute -> quant -> transpose/permute.
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(dims))
        op1 = (
            exir_ops.edge.aten.transpose_copy.int
            if is_op1_transpose
            else exir_ops.edge.aten.permute_copy.default
        )
        node1 = builder.call_operator(
            op=op1,
            args=(x, perm1[0], perm1[1]) if is_op1_transpose else (x, list(perm1)),
        )
        quant_node = self._create_operator(builder, quant_op, node1)
        op2 = (
            exir_ops.edge.aten.transpose_copy.int
            if is_op2_transpose
            else exir_ops.edge.aten.permute_copy.default
        )
        node2 = builder.call_operator(
            op=op2,
            args=(
                (quant_node, perm2[0], perm2[1])
                if is_op2_transpose
                else (quant_node, list(perm2))
            ),
        )
        builder.output([node2])
        gm = builder.get_graph_module()
        expected_op_counts = {
            quant_op: 1,
        }
        expected_op_counts[op1] = 1
        expected_op_counts[op2] = expected_op_counts.get(op2, 0) + 1
        self.check_op_counts(
            gm,
            # pyre-fixme[6]: Incompatible parameter type
            expected_op_counts=expected_op_counts,
        )

        # Check that the pass fuses the two transpose/permute ops.
        fusion_pass_result = FuseTransposeOrPermuteOpPairsPass()(gm)
        self.assertIsNotNone(fusion_pass_result)
        gm_after_pass = fusion_pass_result.graph_module
        if expected_is_fused:
            expected_op_counts[op1] = 0
            expected_op_counts[op2] = 0
        self.check_op_counts(
            gm_after_pass,
            # pyre-fixme[6]: Incompatible parameter type
            expected_op_counts=expected_op_counts,
        )

    def test_fusion_for_forked_transposes(self) -> None:
        # Create a graph with
        # transpose -> quant -> transpose.
        #           -> quant -> transpose.
        #           -> quant -> transpose.
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 3, 4, dtype=torch.float32))
        transpose_node = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(x, 0, 1),
        )
        num_forks = 3
        outputs = []
        for _ in range(num_forks):
            quant_node = builder.call_operator(
                op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                args=(transpose_node, 1.2, 3, 0, 127, torch.int8),
            )
            outputs.append(
                builder.call_operator(
                    op=exir_ops.edge.aten.transpose_copy.int,
                    args=(quant_node, 0, 1),
                )
            )
        builder.output(outputs)
        gm = builder.get_graph_module()
        self.check_op_counts(
            gm,
            expected_op_counts={
                exir_ops.edge.aten.transpose_copy.int: num_forks + 1,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: num_forks,
            },
        )

        # Fuse all the transpose ops.
        p = FuseTransposeOrPermuteOpPairsPass()
        gm_after_pass = cast(PassResult, p(gm)).graph_module
        self.check_op_counts(
            gm_after_pass,
            expected_op_counts={
                exir_ops.edge.aten.transpose_copy.int: 0,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: num_forks,
            },
        )
