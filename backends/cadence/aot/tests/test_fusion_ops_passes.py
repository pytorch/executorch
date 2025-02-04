# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot import compiler
from executorch.backends.cadence.aot.compiler import export_to_edge, quantize_pt2
from executorch.backends.cadence.aot.fuse_ops import (
    FuseFullThenReshapePass,
    FuseMulIntoDequantPass,
    FuseQuantDequantToRequantizePass,
    FuseTransposeOpPairsPass,
)
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch import nn


class TestFusionPassesBase(unittest.TestCase):
    def check_op_counts(
        self,
        graph_module: torch.fx.GraphModule,
        expected_op_counts: dict[EdgeOpOverload, int],
    ) -> None:
        for op, count in expected_op_counts.items():
            self.assertEqual(count_node(graph_module, op), count)


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

        graph_module = FuseQuantDequantToRequantizePass()(graph_module).graph_module
        self.check_op_counts(
            graph_module,
            expected_op_counts={
                # Verify that no dequant/quant pair was replaced with requantize.
                # quantize -> permute -> dequantize should not be replaced with requantize.
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: 1,
                exir_ops.edge.cadence.requantize.default: 0,
            },
        )

    def test_replace_quant_view_dequant_with_requantize(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, 1.2, 3, 0, 127, torch.int8
                )
                x = x.view(-1)
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 4.5, 6, 0, 127, torch.int8
                )
                return x

        inputs = torch.randn(2, 12, 1, 6)
        model = M()
        graph_module = export_to_edge(model, (inputs,)).exported_program().graph_module
        graph_module = FuseQuantDequantToRequantizePass()(graph_module).graph_module
        graph_module.print_readable()

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
        quantized_model = quantize_pt2(model, (inputs,))
        graph_module = (
            export_to_edge(quantized_model, (inputs,)).exported_program().graph_module
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
        class M(torch.nn.Module):
            def forward(self, x):
                x0 = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, 1.5, 0, 0, 255, torch.uint8
                )
                x1 = torch.full([4, 32], 3, dtype=torch.float32)
                x2 = x0 * x1
                return x2

        inputs = (torch.randint(0, 255, [4, 32], dtype=torch.uint8),)
        graph_module = export_to_edge(M(), inputs).exported_program().graph_module
        graph_module = FuseMulIntoDequantPass()(graph_module).graph_module

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
        self.assertEqual(deq_scale, 4.5)

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


class TestFuseTransposeOpPairsPass(TestFusionPassesBase):
    def test_fuse_transpose_pairs(self):
        # Create a graph with transpose -> quant -> transpose.
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 3))
        transpose_node = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(x, 0, 1),
        )
        quant_node = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(transpose_node, 1.2, 3, 0, 127, torch.int8),
        )
        transpose_node = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(quant_node, 0, 1),
        )
        builder.output(transpose_node)
        gm = builder.get_graph_module()
        self.check_op_counts(
            gm,
            expected_op_counts={
                exir_ops.edge.aten.transpose_copy.int: 2,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
            },
        )

        # Check that the pass fuses the two transpose ops.
        gm_after_pass = FuseTransposeOpPairsPass()(gm).graph_module
        self.check_op_counts(
            gm_after_pass,
            expected_op_counts={
                exir_ops.edge.aten.transpose_copy.int: 0,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
            },
        )

    def test_no_fusion_for_transpose_pairs(self):
        # Create a graph with transpose -> quant -> transpose.
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 3, 4))
        transpose_node = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(x, 0, 1),
        )
        quant_node = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(transpose_node, 1.2, 3, 0, 127, torch.int8),
        )
        transpose_node = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(quant_node, 1, 2),
        )
        builder.output(transpose_node)
        gm = builder.get_graph_module()
        self.check_op_counts(
            gm,
            expected_op_counts={
                exir_ops.edge.aten.transpose_copy.int: 2,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
            },
        )

        # No fusion.
        gm_after_pass = FuseTransposeOpPairsPass()(gm).graph_module
        self.check_op_counts(
            gm_after_pass,
            expected_op_counts={
                exir_ops.edge.aten.transpose_copy.int: 2,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: 1,
            },
        )

    def test_fusion_for_forked_transposes(self):
        # Create a graph with transpose -> quant -> transpose.
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

        # Fuse the all the transpose ops.
        gm_after_pass = FuseTransposeOpPairsPass()(gm).graph_module
        self.check_op_counts(
            gm_after_pass,
            expected_op_counts={
                exir_ops.edge.aten.transpose_copy.int: 0,
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: num_forks,
            },
        )
