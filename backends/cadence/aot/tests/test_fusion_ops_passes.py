# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import unittest
from typing import Final, List, Tuple

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot import compiler
from executorch.backends.cadence.aot.compiler import (
    export_to_edge,
    quantize_and_export_to_edge,
)
from executorch.backends.cadence.aot.fuse_ops import (
    FuseFullThenReshapePass,
    FuseMulScalarIntoDequantPass,
    FuseMulTensorIntoDequantPass,
    FuseQuantDequantToRequantizePass,
    FuseTransposeOrPermuteOpPairsPass,
)
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.pass_utils import count_node, op_counts_match
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ProxyValue
from parameterized import parameterized
from torch import nn


class TestFusionPassesBase(unittest.TestCase):
    def check_op_counts(
        self,
        graph_module: torch.fx.GraphModule,
        expected_op_counts: dict[EdgeOpOverload, int],
    ) -> None:
        self.assertTrue(op_counts_match(graph_module, expected_op_counts))


class TestFusionPasses(TestFusionPassesBase):
    def test_addmm_fusion(self):
        class AddmmFeasible1(torch.nn.Module):
            def forward(self, x, y, z):
                t1 = torch.mm(x, y)
                return torch.add(t1, z)

        x = torch.randn(3, 5)
        y = torch.randn(5, 6)
        z = torch.randn(6)

        graph_module = (
            compiler.export_to_cadence(AddmmFeasible1(), (x, y, z))
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()

        # Assert that mm and add were fused to addmm
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.addmm.default), 1)
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.mm.default), 0)
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.add.Tensor), 0)

        class AddmmFeasible2(torch.nn.Module):
            def forward(self, x, y, z):
                t1 = y.view((8, 6))
                t2 = torch.mm(x, t1)
                t3 = t2.view((2, 2, 6))
                return torch.add(t3, z)

        x = torch.randn(4, 8)
        y = torch.randn(2, 4, 6)
        z = torch.randn(6)

        graph_module = (
            compiler.export_to_cadence(AddmmFeasible2(), (x, y, z))
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that mm and add were fused to addmm
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.addmm.default), 1)
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.mm.default), 0)
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.add.Tensor), 0)

        # Bias is a singleton value, broadcastable to output of mm
        class AddmmFeasible3(torch.nn.Module):
            def forward(self, x, y):
                t1 = torch.mm(x, y)
                return torch.add(t1, torch.ones(1))

        x = torch.randn(3, 5)
        y = torch.randn(5, 6)

        graph_module = (
            compiler.export_to_cadence(AddmmFeasible3(), (x, y))
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that mm and add were fused to addmm
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.addmm.default), 1)
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.mm.default), 0)
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.add.Tensor), 0)

        # Bias is not broadcastable to output of mm
        class AddmmInfeasible1(torch.nn.Module):
            def forward(self, x, y, z):
                t1 = y.view((8, 6))
                t2 = torch.mm(x, t1)
                t3 = t2.view((2, 2, 6))
                return torch.add(t3, z)

        x = torch.randn(4, 8)
        y = torch.randn(2, 4, 6)
        z = torch.randn(2, 2, 1)

        graph_module = (
            compiler.export_to_cadence(AddmmInfeasible1(), (x, y, z))
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that mm and add were not fused to addmm, since z cannot be
        # broadcasted to the out of mm.
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.add.Tensor), 1)

        # The add consuming the output of mm has more than one users.
        class AddmmInfeasible2(torch.nn.Module):
            def forward(self, x, y, z):
                t1 = torch.mm(x, y)
                t2 = torch.add(t1, z)
                t3 = torch.add(t2, z)
                return torch.add(t2, t3)

        x = torch.randn(3, 5)
        y = torch.randn(5, 6)
        z = torch.randn(6)

        graph_module = (
            compiler.export_to_cadence(AddmmInfeasible2(), (x, y, z))
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that mm and add were not fused to addmm, since add has multiple
        # users.
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.add.Tensor), 3)

    # TODO(matthiascremon): enable that pass with new flow
    @torch.no_grad()
    @unittest.expectedFailure
    def test_legacy_conv_bn_fusion(self):
        class ModelConvBN(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int, kernel_size: int):
                super().__init__()
                self.conv1d = nn.Conv1d(in_features, out_features, kernel_size)
                self.bn = nn.BatchNorm1d(out_features)

            def forward(self, x):
                y = self.conv1d(x)
                return self.bn(y)

        model = ModelConvBN(64, 1, 2)
        x = torch.randn(1, 64, 4)

        graph_module = (
            compiler.export_to_executorch(model.eval(), (x,))
            .exported_program()
            .exported_program()
            .graph_module
        )
        # Assert that after running the fusion passes, batchnorm was fused with conv1d
        self.assertEqual(
            count_node(graph_module, torch.ops.aten.linear.out)
            + count_node(graph_module, torch.ops.cadence.convolution.out),
            1,
        )
        self.assertEqual(
            count_node(
                graph_module, torch.ops.aten._native_batch_norm_legit_no_training.out
            ),
            0,
        )

    def test_permute_transpose_fusion(self):
        class PermuteTranspose(torch.nn.Module):
            def forward(self, x):
                y = x.permute((0, 2, 4, 1, 3))
                return y.transpose(0, 1)

        x = torch.randn(3, 1, 3, 1, 4)
        graph_module = (
            compiler.export_to_cadence(PermuteTranspose(), (x,))
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that permute op was fused with transpose op
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.permute_copy.default), 1
        )
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.transpose_copy.int), 0
        )

    def test_view_fusion(self):
        class ViewFusion(torch.nn.Module):
            def forward(self, x):
                x = x.view([1, 8, 15])
                x = x.view([1, 1, 120])
                return x.view([1, 12, 10])

        x = torch.randn(8, 5, 3)
        graph_module = (
            compiler.export_to_cadence(ViewFusion(), (x,))
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that only one view op remains
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.view_copy.default), 1
        )

    def test_view_fusion_branched(self):
        class ViewFusion(torch.nn.Module):
            def forward(self, x):
                y = x.view([1, 8, 15])
                z = y.view([1, 1, 120])
                t = y.view([120, 1, 1])
                return z, t

        x = torch.randn(8, 5, 3)
        graph_module = (
            compiler.export_to_cadence(ViewFusion(), (x,))
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # z and t should be fused and y should be eliminated.
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.view_copy.default), 2
        )

    def test_force_quant_dequant_fusion(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, 1.2, 3, 0, 127, torch.int8
                )
                x = torch.permute(x, [2, 0, 1, 3])
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 4.5, 6, 0, 127, torch.int8
                )
                return x

        inputs = torch.randn(2, 12, 1, 6)
        model = M()
        graph_module = export_to_edge(model, (inputs,)).exported_program().graph_module

        graph_module = FuseQuantDequantToRequantizePass(
            force_quant_dequant_fusion=True
        )(graph_module).graph_module
        self.check_op_counts(
            graph_module,
            expected_op_counts={
                # Verify that no dequant/quant pair was replaced with requantize.
                # quantize -> permute -> dequantize should not be replaced with requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 0,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                exir_ops.edge.cadence.requantize.default: 1,
            },
        )

    def test_no_replace_quant_permute_dequant_with_requantize(self):
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
        builder.output(dequant)
        original_graph = builder.get_graph_module()
        converted_graph = FuseQuantDequantToRequantizePass(
            force_quant_dequant_fusion=False
        )(original_graph).graph_module
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                # Verify that no dequant/quant pair was replaced with requantize.
                # quantize -> permute -> dequantize should not be replaced with requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                exir_ops.edge.cadence.requantize.default: 0,
            },
        )

    def test_replace_quant_view_dequant_with_requantize(self):
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
        builder.output(dequant)
        original_graph = builder.get_graph_module()
        converted_graph = FuseQuantDequantToRequantizePass()(
            original_graph
        ).graph_module
        self.check_op_counts(
            converted_graph,
            expected_op_counts={
                # Verify that dequant/quant pair was replaced with requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 0,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                exir_ops.edge.cadence.requantize.default: 1,
            },
        )

    def test_replace_dequant_quant_with_requantize(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 1.2, 3, 0, 127, torch.int8
                )
                x = torch.permute(x, [2, 0, 1, 3])
                x = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, 4.5, 6, 0, 127, torch.int8
                )
                return x

        inputs = torch.randn(2, 12, 1, 6).to(torch.int8)
        model = M()
        graph_module = export_to_edge(model, (inputs,)).exported_program().graph_module
        graph_module = FuseQuantDequantToRequantizePass()(graph_module).graph_module

        self.check_op_counts(
            graph_module,
            expected_op_counts={
                # Verify that dequant -> permute -> quant was replaced with permute -> requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 0,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                exir_ops.edge.cadence.requantize.default: 1,
            },
        )

    def test_replace_dequant_permute_quant_with_requantize(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 1.2, 3, 0, 127, torch.int8
                )
                x = torch.permute(x, [2, 0, 1, 3])
                x = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, 4.5, 6, 0, 127, torch.int8
                )
                return x

        inputs = torch.randn(2, 12, 1, 6).to(torch.int8)
        model = M()
        graph_module = export_to_edge(model, (inputs,)).exported_program().graph_module
        graph_module = FuseQuantDequantToRequantizePass()(graph_module).graph_module

        self.check_op_counts(
            graph_module,
            expected_op_counts={
                # Verify that dequant -> permute -> quant was replaced with permute -> requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 0,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 0,
                exir_ops.edge.cadence.requantize.default: 1,
            },
        )

    def test_remove_nop_dequant_quant(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.lin1 = torch.nn.Linear(6, 12, bias=False)
                self.lin2 = torch.nn.Linear(12, 24, bias=False)

            def forward(self, x):
                x = self.lin1(x)
                # redundant dequant+quant will be created around this permute
                x = torch.permute(x, [0, 2, 1, 3])
                x = self.lin2(x)
                return x

        inputs = torch.randn(2, 12, 1, 6)
        model = M()
        graph_module = (
            quantize_and_export_to_edge(model, (inputs,))
            .exported_program()
            .graph_module
        )
        graph_module = FuseQuantDequantToRequantizePass()(graph_module).graph_module
        self.check_op_counts(
            graph_module,
            expected_op_counts={
                # Verify that one dequant/quant pair was removed
                # Expect 1 quantize ops: 1 input
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
                # Expect 1 dequant op at the end (output of second linear)
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
            },
        )

    def test_fuse_mul_into_dequant(self):
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
        builder.output(mul)
        graph_module = FuseMulTensorIntoDequantPass()(
            builder.get_graph_module()
        ).graph_module

        # verify that the mul and full ops were removed
        self.check_op_counts(
            graph_module,
            expected_op_counts={
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                exir_ops.edge.aten.full.default: 0,
                exir_ops.edge.aten.mul.Tensor: 0,
            },
        )

        # verify that the dequant scale value was updated correctly
        for node in graph_module.graph.nodes:
            if (
                node.target
                == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ):
                deq_scale = node.args[1]
        self.assertEqual(deq_scale, DEQUANT_SCALE * FULL_VALUE)

    def test_fuse_mul_scalar_into_dequant(self):
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
        builder.output(mul_scalar)
        graph_module = builder.get_graph_module()

        graph_module = FuseMulScalarIntoDequantPass()(graph_module).graph_module

        # verify that the mul and full ops were removed
        self.check_op_counts(
            graph_module,
            expected_op_counts={
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                exir_ops.edge.aten.mul.Scalar: 0,
            },
        )

        # verify that the dequant scale value was updated correctly
        for node in graph_module.graph.nodes:
            if (
                node.target
                == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ):
                deq_scale = node.args[1]
        self.assertEqual(deq_scale, dequant_scale * mul_value)

    def test_fuse_then_transpose_pass(self):
        # Create a graph with full -> transpose.
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
        builder.output(view_node)
        gm = builder.get_graph_module()
        self.check_op_counts(
            gm,
            expected_op_counts={
                exir_ops.edge.aten.full.default: 1,
                exir_ops.edge.aten.transpose_copy.int: 1,
                exir_ops.edge.aten.permute_copy.default: 1,
                exir_ops.edge.aten.view_copy.default: 1,
            },
        )

        # Check that the pass fuses the full with all other ops (transpose, permute, view).
        gm_after_pass = FuseFullThenReshapePass()(gm).graph_module
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

    @parameterized.expand(
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
    ):
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

    def test_fusion_for_forked_transposes(self):
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
        gm_after_pass = FuseTransposeOrPermuteOpPairsPass()(gm).graph_module
        self.check_op_counts(
            gm_after_pass,
            expected_op_counts={
                exir_ops.edge.aten.transpose_copy.int: 0,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: num_forks,
            },
        )
