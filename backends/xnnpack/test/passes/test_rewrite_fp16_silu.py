# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class TestRewriteFp16SiluPass(unittest.TestCase):
    edge_mul = "executorch_exir_dialects_edge__ops_aten_mul_Tensor"
    edge_sigmoid = "executorch_exir_dialects_edge__ops_aten_sigmoid_default"
    edge_silu = "executorch_exir_dialects_edge__ops_aten_silu_default"

    class Silu(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    def setUp(self):
        torch._dynamo.reset()

    def _export_silu(self, dtype):
        return torch.export.export(
            self.Silu(),
            (torch.randn(2, 3, dtype=dtype),),
            strict=True,
        )

    def _get_preserved_edge_fp16_silu(self):
        edge_program = to_edge(
            self._export_silu(torch.float16),
            compile_config=get_xnnpack_edge_compile_config(),
        ).exported_program()
        graph = edge_program.graph_module.graph
        input_node = next(node for node in graph.nodes if node.op == "placeholder")
        output_node = next(node for node in graph.nodes if node.op == "output")
        silu = output_node.args[0][0]
        silu.target = exir_ops.edge.aten.silu.default
        silu.args = (input_node,)
        silu.kwargs = {}
        graph.eliminate_dead_code()
        graph.lint()
        edge_program.graph_module.recompile()
        return edge_program

    def test_ops_to_not_decompose_filters_for_fp16(self):
        partitioner = XnnpackPartitioner()

        fp16_program = self._export_silu(torch.float16)
        preserved_ops, filter_fn = partitioner.ops_to_not_decompose(fp16_program)
        fp16_silu = next(
            node
            for node in fp16_program.graph.nodes
            if node.target == torch.ops.aten.silu.default
        )
        self.assertIn(torch.ops.aten.silu.default, preserved_ops)
        self.assertIsNotNone(filter_fn)
        self.assertTrue(filter_fn(fp16_silu))

        fp32_program = self._export_silu(torch.float32)
        _, filter_fn = partitioner.ops_to_not_decompose(fp32_program)
        fp32_silu = next(
            node
            for node in fp32_program.graph.nodes
            if node.target == torch.ops.aten.silu.default
        )
        self.assertIsNotNone(filter_fn)
        self.assertFalse(filter_fn(fp32_silu))

    def test_preprocess_rewrites_preserved_fp16_silu(self):
        result = XNNPACKPassManager(self._get_preserved_edge_fp16_silu()).transform()
        targets = [node.target for node in result.graph.nodes]

        self.assertNotIn(exir_ops.edge.aten.silu.default, targets)
        self.assertEqual(targets.count(exir_ops.edge.aten.sigmoid.default), 1)
        self.assertEqual(targets.count(exir_ops.edge.aten.mul.Tensor), 1)

    def test_fp32_silu_uses_default_decomposition(self):
        (
            Tester(self.Silu(), (torch.randn(2, 3),))
            .export()
            .to_edge()
            .check_count({self.edge_silu: 0, self.edge_sigmoid: 1, self.edge_mul: 1})
        )

    def test_to_edge_transform_and_lower_delegates_fp16_silu(self):
        (
            Tester(self.Silu(), (torch.randn(2, 3, dtype=torch.float16),))
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not([self.edge_silu])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
