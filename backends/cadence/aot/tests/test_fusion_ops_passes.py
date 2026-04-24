# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import copy
import operator
import unittest
from typing import cast, Final, List, Tuple

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.fuse_ops import (
    FuseBatchNormWithConv,
    FuseCascadedTransposeOrPermuteOps,
    FuseCascadedViewOps,
    FuseFullThenReshapePass,
    FuseMeanKeepDimWithViewPass,
    FuseMMWithAdd,
    FuseMulScalarIntoDequantPass,
    FuseMulTensorIntoDequantPass,
    FuseMulTensorIntoQuantPass,
    FuseQuantDequantToRequantizePass,
    FuseQuantizedBatchNormWithConv,
    FuseTransposeOrPermuteOpPairsPass,
    HierarchicalCSEPass,
)
from executorch.backends.cadence.aot.pass_utils import count_node, op_counts_match
from executorch.backends.cadence.aot.typing_stubs import expand
from executorch.backends.test.graph_builder import GraphBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult, ProxyValue

from parameterized import parameterized
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

    def test_cascaded_permutes_multiple_users(self) -> None:
        # Test case where intermediate permute has multiple users.
        #            x
        #            |
        #         permute1
        #       /    |     \
        # permute2 permute3 permute4
        #    |       |         |
        #   out0    out1    permute5
        #                      |
        #                     out2

        builder = GraphBuilder()
        x_input = torch.randn(2, 3, 8, 8, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        permute1 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 2, 3, 1]),
        )
        permute2 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(permute1, [0, 3, 1, 2]),
        )
        permute3 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(permute1, [0, 1, 3, 2]),
        )
        permute4 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(permute1, [3, 2, 1, 0]),
        )
        permute5 = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(permute4, [1, 2, 3, 0]),
        )
        builder.output([permute2, permute3, permute5])
        original_graph = builder.get_graph_module()
        graph_copy = copy.deepcopy(original_graph)

        p = FuseCascadedTransposeOrPermuteOps()
        result = p.call(original_graph)
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # permute2 becomes a no-op, permute3 and permute5 fused with preceding permutes
        # into new single permutes.
        output0, output1, output2 = converted_graph.graph.output_node().args[0]
        # out0: permute1 + permute2 = identity, so it connects to the graph input.
        graph_input = converted_graph.graph.find_nodes(op="placeholder")[0]
        self.assertIs(output0, graph_input)
        # out1: permute1 [0,2,3,1] + permute3 [0,1,3,2] fused to [0,2,1,3].
        self.assertEqual(output1.target, exir_ops.edge.aten.permute_copy.default)
        self.assertIs(output1.args[0], graph_input)
        self.assertEqual(output1.args[1], [0, 2, 1, 3])
        # out2: permute1 [0,2,3,1] + permute4 [3,2,1,0] + permute5 [1,2,3,0]
        # fused to [3,2,0,1].
        self.assertEqual(output2.target, exir_ops.edge.aten.permute_copy.default)
        self.assertIs(output2.args[0], graph_input)
        self.assertEqual(output2.args[1], [3, 2, 0, 1])
        validate_numerics(
            graph_copy,
            converted_graph,
            (x_input,),
            "FuseCascadedTransposeOrPermuteOps_multiple_users",
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
        x_input = torch.randint(low=0, high=255, size=INPUT_SHAPE, dtype=torch.uint8)
        x = builder.placeholder("x", x_input)
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
        gm_before = copy.deepcopy(original_graph)

        p = FuseMulTensorIntoDequantPass()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Validate numerical accuracy
        validate_numerics(
            gm_before, converted_graph, (x_input,), "FuseMulTensorIntoDequantPass"
        )

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

    def test_fuse_mul_into_dequant_no_match(self) -> None:
        """
        Test that FuseMulTensorIntoDequantPass does NOT modify the graph
        when the mul node's inputs are not dequant + full.
        """
        INPUT_SHAPE: Final[List[int]] = [4, 32]

        builder = GraphBuilder()
        # Create two regular placeholder inputs (not dequant outputs)
        x_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)
        y_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        y = builder.placeholder("y", y_input)

        # Mul of two placeholders - no dequant node involved
        mul = builder.call_operator(
            op=exir_ops.edge.aten.mul.Tensor,
            args=(x, y),
        )
        builder.output([mul])
        original_graph = builder.get_graph_module()

        p = FuseMulTensorIntoDequantPass()
        result = cast(PassResult, p(original_graph))

        # The pass should NOT modify the graph since there's no dequant node
        self.assertFalse(result.modified)

        # Verify that the mul op is still present
        self.check_op_counts(
            result.graph_module,
            expected_op_counts={
                exir_ops.edge.aten.mul.Tensor: 1,
            },
        )

    def test_fuse_mul_scalar_into_dequant(self) -> None:
        dequant_scale = 0.006
        mul_value = 0.3

        builder = GraphBuilder()
        x_input = torch.randn(2, 3, 4, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
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
        gm_before = copy.deepcopy(original_graph)

        p = FuseMulScalarIntoDequantPass()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Validate numerical accuracy
        validate_numerics(
            gm_before, converted_graph, (x_input,), "FuseMulScalarIntoDequantPass"
        )

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
        x_input = torch.randn(4, 32, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
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
        gm_before = copy.deepcopy(original_graph)

        p = FuseMulTensorIntoQuantPass()
        result = cast(PassResult, p(original_graph))
        self.assertTrue(result.modified)
        converted_graph = result.graph_module

        # Validate numerical accuracy
        validate_numerics(
            gm_before, converted_graph, (x_input,), "FuseMulTensorIntoQuantPass"
        )

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


class TestHierarchicalCSEPass(TestFusionPassesBase):
    """Tests for HierarchicalCSEPass that performs CSE across all submodules.

    The HierarchicalCSEPass eliminates redundant computations (common subexpressions)
    at all levels of the module hierarchy, including nested subgraphs.
    """

    # -------------------------------------------------------------------------
    # Graph Creation Utilities
    # -------------------------------------------------------------------------

    def _create_duplicate_add_scalar_graph(
        self, shape: tuple[int, ...] = (8, 8)
    ) -> torch.fx.GraphModule:
        """Create a graph with two identical add.Scalar operations.

        Graph structure:
            x (placeholder)
            ├── add.Scalar(x, 1)  ─┐
            └── add.Scalar(x, 1)  ─┴── add.Tensor (result)
        """
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape))
        add1 = builder.call_operator(exir_ops.edge.aten.add.Scalar, (x, 1))
        add2 = builder.call_operator(exir_ops.edge.aten.add.Scalar, (x, 1))
        result = builder.call_operator(exir_ops.edge.aten.add.Tensor, (add1, add2))
        builder.output([result])
        return builder.get_graph_module()

    def _create_different_add_scalar_graph(
        self, shape: tuple[int, ...] = (8, 8)
    ) -> torch.fx.GraphModule:
        """Create a graph with add.Scalar operations using different values.

        Graph structure:
            x (placeholder)
            ├── add.Scalar(x, 1)  ─┐
            ├── add.Scalar(x, 2)  ─┼── add.Tensor chain (result)
            └── add.Scalar(x, 3)  ─┘
        """
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape))
        add1 = builder.call_operator(exir_ops.edge.aten.add.Scalar, (x, 1))
        add2 = builder.call_operator(exir_ops.edge.aten.add.Scalar, (x, 2))
        add3 = builder.call_operator(exir_ops.edge.aten.add.Scalar, (x, 3))
        temp = builder.call_operator(exir_ops.edge.aten.add.Tensor, (add1, add2))
        result = builder.call_operator(exir_ops.edge.aten.add.Tensor, (temp, add3))
        builder.output([result])
        return builder.get_graph_module()

    def _create_diamond_pattern_graph(
        self, shape: tuple[int, ...] = (32, 64)
    ) -> torch.fx.GraphModule:
        """Create a diamond-shaped graph with duplicate and unique operations.

        Graph structure:
            x (placeholder)
            ├── add.Scalar(x, 5)  ─── mul.Scalar(_, 2)  ─┐
            └── add.Scalar(x, 5)  ─── mul.Scalar(_, 3)  ─┴── add.Tensor (result)
        """
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape))
        add_branch1 = builder.call_operator(exir_ops.edge.aten.add.Scalar, (x, 5))
        add_branch2 = builder.call_operator(exir_ops.edge.aten.add.Scalar, (x, 5))
        mul1 = builder.call_operator(exir_ops.edge.aten.mul.Scalar, (add_branch1, 2))
        mul2 = builder.call_operator(exir_ops.edge.aten.mul.Scalar, (add_branch2, 3))
        result = builder.call_operator(exir_ops.edge.aten.add.Tensor, (mul1, mul2))
        builder.output([result])
        return builder.get_graph_module()

    def _create_map_body_with_duplicate_ops(
        self, sample_inp: torch.Tensor
    ) -> torch.fx.GraphModule:
        """Create a map function body with duplicate add.Scalar operations."""
        builder = GraphBuilder()
        x = builder.placeholder("x", sample_inp)
        add1 = builder.call_operator(torch.ops.aten.add.Scalar, (x, 1))
        add2 = builder.call_operator(torch.ops.aten.add.Scalar, (x, 1))
        result = builder.call_operator(torch.ops.aten.add.Tensor, (add1, add2))
        builder.output([result])
        return builder.get_graph_module()

    def _create_map_body_with_mixed_ops(
        self, sample_inp: torch.Tensor
    ) -> torch.fx.GraphModule:
        """Create a map function body with duplicate adds and different muls."""
        builder = GraphBuilder()
        x = builder.placeholder("x", sample_inp)
        add1 = builder.call_operator(torch.ops.aten.add.Scalar, (x, 1))
        add2 = builder.call_operator(torch.ops.aten.add.Scalar, (x, 1))
        mul1 = builder.call_operator(torch.ops.aten.mul.Scalar, (add1, 2))
        mul2 = builder.call_operator(torch.ops.aten.mul.Scalar, (add2, 3))
        result = builder.call_operator(torch.ops.aten.add.Tensor, (mul1, mul2))
        builder.output([result])
        return builder.get_graph_module()

    def _create_map_impl_graph(
        self,
        map_body: torch.fx.GraphModule,
        batch_size: int = 4,
        feature_size: int = 8,
    ) -> torch.fx.GraphModule:
        """Wrap a map body function in a map_impl graph."""
        inp = torch.randn(batch_size, feature_size)
        builder = GraphBuilder()
        inp_proxy = builder.placeholder("inp", inp)
        map_result = builder.call_operator(
            torch.ops.higher_order.map_impl, (map_body, (inp_proxy,), ())
        )
        map_getitem = builder.call_getitem(map_result, 0)
        builder.output([map_getitem])
        return builder.get_graph_module()

    def _get_map_body(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Extract the map body submodule from a graph containing map_impl."""
        map_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.map_impl
        )
        self.assertEqual(len(map_nodes), 1, "Should have exactly one map_impl node")
        map_body_getattr = map_nodes[0].args[0]
        self.assertTrue(hasattr(gm, map_body_getattr.target))
        map_body = getattr(gm, map_body_getattr.target)
        self.assertIsInstance(map_body, torch.fx.GraphModule)
        return cast(torch.fx.GraphModule, map_body)

    def _apply_cse_pass(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Apply HierarchicalCSEPass and return the resulting graph module."""
        p = HierarchicalCSEPass()
        return cast(PassResult, p(gm)).graph_module

    # -------------------------------------------------------------------------
    # Test Cases
    # -------------------------------------------------------------------------

    def test_cse_removes_duplicate_add_scalar(self) -> None:
        """Test that CSE removes duplicate add.Scalar operations with same input."""
        gm = self._create_duplicate_add_scalar_graph()

        self.assertEqual(
            count_node(gm, exir_ops.edge.aten.add.Scalar),
            2,
            "Should have 2 duplicate add.Scalar before CSE",
        )

        gm_after = self._apply_cse_pass(gm)

        self.assertEqual(
            count_node(gm_after, exir_ops.edge.aten.add.Scalar),
            1,
            "CSE should have eliminated duplicate add.Scalar operation",
        )

    def test_cse_with_map_impl_duplicate_ops(self) -> None:
        """Test CSE on a program with map_impl containing duplicate operations."""
        sample_inp = torch.randn(8)
        map_body = self._create_map_body_with_duplicate_ops(sample_inp)
        gm = self._create_map_impl_graph(map_body)

        # Verify before CSE
        map_body_before = self._get_map_body(gm)
        self.assertEqual(
            count_node(map_body_before, torch.ops.aten.add.Scalar),
            2,
            "Map body should have 2 duplicate add.Scalar ops before CSE",
        )

        # Apply CSE
        gm_after = self._apply_cse_pass(gm)

        # Verify after CSE
        map_body_after = self._get_map_body(gm_after)
        self.assertEqual(
            count_node(map_body_after, torch.ops.aten.add.Scalar),
            1,
            "CSE should have eliminated duplicate add.Scalar in map body",
        )

    def test_cse_with_map_impl_mixed_duplicate_and_unique_ops(self) -> None:
        """Test CSE on map_impl with both duplicate and unique operations."""
        sample_inp = torch.randn(8)
        map_body = self._create_map_body_with_mixed_ops(sample_inp)
        gm = self._create_map_impl_graph(map_body)

        # Verify before CSE
        map_body_before = self._get_map_body(gm)
        self.assertEqual(
            count_node(map_body_before, torch.ops.aten.add.Scalar),
            2,
            "Should have 2 duplicate add.Scalar before CSE",
        )
        self.assertEqual(
            count_node(map_body_before, torch.ops.aten.mul.Scalar),
            2,
            "Should have 2 different mul.Scalar before CSE",
        )

        # Apply CSE
        gm_after = self._apply_cse_pass(gm)

        # Verify after CSE
        map_body_after = self._get_map_body(gm_after)
        self.assertEqual(
            count_node(map_body_after, torch.ops.aten.add.Scalar),
            1,
            "CSE should have merged duplicate add.Scalar to 1",
        )
        self.assertEqual(
            count_node(map_body_after, torch.ops.aten.mul.Scalar),
            2,
            "CSE should NOT merge different mul.Scalar operations",
        )

    def test_cse_preserves_different_operations(self) -> None:
        """Test that CSE does not eliminate operations with different arguments."""
        gm = self._create_different_add_scalar_graph()

        self.assertEqual(
            count_node(gm, exir_ops.edge.aten.add.Scalar),
            3,
            "Should have 3 different add.Scalar before CSE",
        )

        gm_after = self._apply_cse_pass(gm)

        self.assertEqual(
            count_node(gm_after, exir_ops.edge.aten.add.Scalar),
            3,
            "CSE should NOT eliminate add.Scalar ops with different scalar values",
        )

    def test_cse_diamond_pattern(self) -> None:
        """Test CSE on diamond-shaped graph where ops share inputs."""
        gm = self._create_diamond_pattern_graph()

        self.check_op_counts(
            gm,
            expected_op_counts={
                exir_ops.edge.aten.add.Scalar: 2,
                exir_ops.edge.aten.mul.Scalar: 2,
            },
        )

        gm_after = self._apply_cse_pass(gm)

        self.check_op_counts(
            gm_after,
            expected_op_counts={
                exir_ops.edge.aten.add.Scalar: 1,  # Merged to one
                exir_ops.edge.aten.mul.Scalar: 2,  # Still two (different args)
            },
        )


class TestFuseBatchNormWithConv(unittest.TestCase):
    """Tests for FuseBatchNormWithConv pass."""

    def test_pass_runs_without_errors(self) -> None:
        """Test that the pass can run on a graph without errors.

        Note: This test uses placeholder nodes for weights instead of get_attr nodes,
        so no actual fusion will occur. This test verifies the pass code compiles
        and runs correctly. Full integration testing with real models should verify
        the actual fusion behavior.
        """
        builder = GraphBuilder()

        # Create input tensor: (N=1, C=3, H=4, W=4)
        x_tensor = torch.randn([1, 3, 4, 4], dtype=torch.float32)
        x = builder.placeholder("x", x_tensor)

        # Create convolution weights: (out_channels=3, in_channels=3, kH=3, kW=3)
        weight_tensor = torch.randn([3, 3, 3, 3], dtype=torch.float32)
        weight = builder.placeholder("weight", weight_tensor)

        # Create convolution bias
        bias_tensor = torch.randn([3], dtype=torch.float32)
        bias = builder.placeholder("bias", bias_tensor)

        # Create convolution node
        conv = builder.call_operator(
            op=exir_ops.edge.aten.convolution.default,
            args=(
                x,
                weight,
                bias,
                [1, 1],  # stride
                [1, 1],  # padding
                [1, 1],  # dilation
                False,  # transposed
                [0, 0],  # output_padding
                1,  # groups
            ),
        )

        # Create batch_norm parameters
        bn_weight_tensor = torch.ones([3], dtype=torch.float32)
        bn_weight = builder.placeholder("bn_weight", bn_weight_tensor)
        bn_bias_tensor = torch.zeros([3], dtype=torch.float32)
        bn_bias = builder.placeholder("bn_bias", bn_bias_tensor)
        running_mean_tensor = torch.zeros([3], dtype=torch.float32)
        running_mean = builder.placeholder("running_mean", running_mean_tensor)
        running_var_tensor = torch.ones([3], dtype=torch.float32)
        running_var = builder.placeholder("running_var", running_var_tensor)

        # Create batch_norm node
        bn = builder.call_operator(
            op=exir_ops.edge.aten.native_batch_norm.default,
            args=(
                conv,
                bn_weight,
                bn_bias,
                running_mean,
                running_var,
                False,  # training
                0.1,  # momentum
                1e-5,  # eps
            ),
        )

        # Get first element of batch_norm output tuple
        getitem = builder.call_operator(
            op=operator.getitem,
            args=(bn, 0),
        )

        builder.output([getitem])
        gm = builder.get_graph_module()

        # Verify initial state: has both convolution and batch_norm
        self.assertEqual(count_node(gm, exir_ops.edge.aten.convolution.default), 1)
        self.assertEqual(
            count_node(gm, exir_ops.edge.aten.native_batch_norm.default), 1
        )

        # Run the fusion pass - should run without errors
        p = FuseBatchNormWithConv()
        result = cast(PassResult, p(gm))

        # Verify pass returns a valid PassResult
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.graph_module)
        # Note: modified is False because weights are placeholders, not get_attr nodes.
        # The pass only fuses when weights are registered module parameters.
        self.assertFalse(result.modified)

        # Verify nodes are unchanged after pass (no fusion occurred due to placeholder weights)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.convolution.default), 1
        )
        self.assertEqual(
            count_node(
                result.graph_module, exir_ops.edge.aten.native_batch_norm.default
            ),
            1,
        )


class TestFuseQuantizedBatchNormWithConv(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "conv1d_bn1d",
                exir_ops.edge.quantized.conv1d.default,
                exir_ops.edge.quantized.batch_norm1d.default,
                exir_ops.edge.quantized.conv1d_prepack,
                (1, 3, 10),  # input shape: (N, C, L)
                (3, 3, 3),  # weight shape: (out_channels, in_channels, kernel)
                [1],  # stride
                [1],  # padding
                [1],  # dilation
            ),
            (
                "conv2d_bn2d",
                exir_ops.edge.quantized.conv2d.new,
                exir_ops.edge.quantized.batch_norm2d.default,
                exir_ops.edge.quantized.conv2d_prepack,
                (1, 3, 8, 8),  # input shape: (N, C, H, W)
                (3, 3, 3, 3),  # weight shape: (out_channels, in_channels, kH, kW)
                [1, 1],  # stride
                [1, 1],  # padding
                [1, 1],  # dilation
            ),
        ]
    )
    def test_fuse_quantized_conv_bn(
        self,
        _name: str,
        conv_op: EdgeOpOverload,
        bn_op: EdgeOpOverload,
        prepack_op: EdgeOpOverload,
        _input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
    ) -> None:
        """
        Test that FuseQuantizedBatchNormWithConv pass fuses quantized conv + bn
        into just the quantized conv op with fused weights.
        """
        out_channels = weight_shape[0]
        scale = 0.1
        zero_point = 0
        eps = 1e-5
        groups = 1

        # Create a quantized weight tensor
        weight_fp = torch.randn(weight_shape, dtype=torch.float32)
        weight_quant = torch.quantize_per_tensor(
            weight_fp, scale=0.1, zero_point=0, dtype=torch.qint8
        )
        bias = torch.randn(out_channels, dtype=torch.float32)

        # Create packed params using actual prepack op
        packed_params = prepack_op(
            weight_quant, bias, stride, padding, dilation, groups
        )

        # Create batch norm parameters
        # Note: Using schema parameter names 'mean' and 'var' to match
        # quantized::batch_norm1d schema, not 'running_mean'/'running_var'
        bn_weight = torch.ones(out_channels, dtype=torch.float32)
        bn_bias = torch.zeros(out_channels, dtype=torch.float32)
        mean = torch.zeros(out_channels, dtype=torch.float32)
        var = torch.ones(out_channels, dtype=torch.float32)

        # Create a root module with registered attributes
        class RootModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.packed_params = packed_params
                self.register_buffer("bn_weight", bn_weight)
                self.register_buffer("bn_bias", bn_bias)
                self.register_buffer("mean", mean)
                self.register_buffer("var", var)

        root = RootModule()

        # Manually build the graph
        graph = torch.fx.Graph()

        # Create input placeholder
        x_node = graph.placeholder("x")

        # Create get_attr nodes for packed params and bn params
        packed_params_node = graph.get_attr("packed_params")
        bn_weight_node = graph.get_attr("bn_weight")
        bn_bias_node = graph.get_attr("bn_bias")
        mean_node = graph.get_attr("mean")
        var_node = graph.get_attr("var")

        # Create quantized conv node: (input, packed_params, scale, zero_point)
        conv_node = graph.call_function(
            conv_op,
            args=(x_node, packed_params_node, scale, zero_point),
        )

        # Create quantized batch_norm node:
        # (input, weight, bias, mean, var, eps, scale, zero_point)
        bn_node = graph.call_function(
            bn_op,
            args=(
                conv_node,
                bn_weight_node,
                bn_bias_node,
                mean_node,
                var_node,
                eps,
                scale,
                zero_point,
            ),
        )

        # Output the batch_norm result
        graph.output(bn_node)

        # Create GraphModule with the root module
        gm = torch.fx.GraphModule(root, graph)

        # Verify initial graph has both conv and bn
        self.assertEqual(count_node(gm, conv_op), 1)
        self.assertEqual(count_node(gm, bn_op), 1)

        # Test the fusion logic directly via maybe_remove_or_replace
        # This avoids the recompile step which has serialization issues with ScriptObjects
        p = FuseQuantizedBatchNormWithConv()

        # Find the conv node and call maybe_remove_or_replace
        for node in gm.graph.nodes:
            if node.target == conv_op:
                result = p.maybe_remove_or_replace(node)
                self.assertTrue(
                    result, "Fusion should succeed for quantized conv+bn pattern"
                )
                break

        # Verify fusion occurred: bn should be removed, conv remains
        self.assertEqual(count_node(gm, conv_op), 1)
        self.assertEqual(count_node(gm, bn_op), 0)


class TestFuseMeanKeepDimWithViewPass(TestFusionPassesBase):
    def test_keepdim_true_to_false(self) -> None:
        """mean(keepdim=True) + view that squeezes reduction dims → mean(keepdim=False)."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 62, 4, 4))
        mean = builder.call_operator(
            op=exir_ops.edge.aten.mean.dim, args=(x, [-1, -2], True)
        )
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mean, [2, 62])
        )
        builder.output([view])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = cast(PassResult, FuseMeanKeepDimWithViewPass()(original))
        gm = result.graph_module
        self.assertTrue(result.modified)

        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 0)
        mean_nodes = gm.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.mean.dim
        )
        self.assertEqual(len(mean_nodes), 1)
        self.assertFalse(mean_nodes[0].args[2])

        validate_numerics(
            gm_before, gm, (torch.randn(2, 62, 4, 4),), "FuseMeanKeepDimWithViewPass"
        )

    def test_keepdim_false_to_true(self) -> None:
        """mean(keepdim=False) + view that unsqueezes at reduction dims → mean(keepdim=True)."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 62, 4, 4))
        mean = builder.call_operator(
            op=exir_ops.edge.aten.mean.dim, args=(x, [-1, -2], False)
        )
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mean, [2, 62, 1, 1])
        )
        builder.output([view])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = cast(PassResult, FuseMeanKeepDimWithViewPass()(original))
        gm = result.graph_module
        self.assertTrue(result.modified)

        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 0)
        mean_nodes = gm.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.mean.dim
        )
        self.assertEqual(len(mean_nodes), 1)
        self.assertTrue(mean_nodes[0].args[2])

        validate_numerics(
            gm_before, gm, (torch.randn(2, 62, 4, 4),), "FuseMeanKeepDimWithViewPass"
        )

    def test_keepdim_true_view_does_not_match(self) -> None:
        """View reshapes to something other than squeezing reduction dims → no change."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 62, 4, 4))
        mean = builder.call_operator(
            op=exir_ops.edge.aten.mean.dim, args=(x, [-1, -2], True)
        )
        # Reshape to a different layout, not a simple squeeze of reduction dims.
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mean, [1, 2, 62])
        )
        builder.output([view])
        original = builder.get_graph_module()

        result = cast(PassResult, FuseMeanKeepDimWithViewPass()(original))
        self.assertFalse(result.modified)

    def test_keepdim_false_view_wrong_unsqueeze(self) -> None:
        """View inserts 1s at wrong positions → no change."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 62, 4, 4))
        mean = builder.call_operator(
            op=exir_ops.edge.aten.mean.dim, args=(x, [-1, -2], False)
        )
        # 1s at positions 0 and 1 instead of 2 and 3.
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mean, [1, 1, 2, 62])
        )
        builder.output([view])
        original = builder.get_graph_module()

        result = cast(PassResult, FuseMeanKeepDimWithViewPass()(original))
        self.assertFalse(result.modified)

    def test_mean_multiple_users_no_change(self) -> None:
        """Mean has multiple users → no change."""
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 62, 4, 4))
        mean = builder.call_operator(
            op=exir_ops.edge.aten.mean.dim, args=(x, [-1, -2], True)
        )
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mean, [2, 62])
        )
        neg = builder.call_operator(op=exir_ops.edge.aten.neg.default, args=(mean,))
        builder.output([view, neg])
        original = builder.get_graph_module()

        result = cast(PassResult, FuseMeanKeepDimWithViewPass()(original))
        self.assertFalse(result.modified)

    def test_reduce_single_dim(self) -> None:
        """Reduction over a single dim, both directions."""
        # keepdim=True → False
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(3, 4, 5))
        mean = builder.call_operator(
            op=exir_ops.edge.aten.mean.dim, args=(x, [1], True)
        )
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mean, [3, 5])
        )
        builder.output([view])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = cast(PassResult, FuseMeanKeepDimWithViewPass()(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.view_copy.default), 0
        )

        validate_numerics(
            gm_before,
            result.graph_module,
            (torch.randn(3, 4, 5),),
            "FuseMeanKeepDimWithViewPass",
        )

        # keepdim=False → True
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(3, 4, 5))
        mean = builder.call_operator(
            op=exir_ops.edge.aten.mean.dim, args=(x, [1], False)
        )
        view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default, args=(mean, [3, 1, 5])
        )
        builder.output([view])
        original = builder.get_graph_module()
        gm_before = copy.deepcopy(original)

        result = cast(PassResult, FuseMeanKeepDimWithViewPass()(original))
        self.assertTrue(result.modified)
        self.assertEqual(
            count_node(result.graph_module, exir_ops.edge.aten.view_copy.default), 0
        )

        validate_numerics(
            gm_before,
            result.graph_module,
            (torch.randn(3, 4, 5),),
            "FuseMeanKeepDimWithViewPass",
        )
