# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch
from executorch.backends.xnnpack._passes.lift_constant_scalar_operands_pass import (
    LiftConstantScalarOperandsPass,
)
from executorch.backends.xnnpack.partition.graphs import sdpa
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class TestLiftConstantScalarOperandsPass(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class MulScalar(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.mul.Scalar(x, 0.5)

    class AddScalar(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.add.Scalar(x, 0.5)

    def _to_edge_graph(self, module):
        edge = to_edge(
            torch.export.export(module, (torch.randn(2, 3),), strict=True),
            compile_config=get_xnnpack_edge_compile_config(skip_dim_order=True),
        )
        return edge.transform([LiftConstantScalarOperandsPass()]).exported_program()

    def test_lifts_mul_scalar_operand(self):
        graph = self._to_edge_graph(self.MulScalar()).graph_module.graph

        self.assertFalse(
            any(node.target == exir_ops.edge.aten.mul.Scalar for node in graph.nodes)
        )
        self.assertTrue(
            any(node.target == exir_ops.edge.aten.mul.Tensor for node in graph.nodes)
        )
        self.assertTrue(any(node.op == "get_attr" for node in graph.nodes))

    def test_keeps_unmapped_scalar_op(self):
        graph = self._to_edge_graph(self.AddScalar()).graph_module.graph

        self.assertTrue(
            any(node.target == exir_ops.edge.aten.add.Scalar for node in graph.nodes)
        )

    def test_keeps_sdpa_scale_mul_scalar(self):
        graph_module = deepcopy(sdpa.get_graphs()[0])

        LiftConstantScalarOperandsPass()(graph_module)

        scale_mul_count = 0
        lifted_mul_count = 0
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == exir_ops.edge.aten.mul.Scalar:
                scale_mul_count += 1
            if node.target == exir_ops.edge.aten.mul.Tensor:
                lifted_mul_count += 1

        self.assertEqual(scale_mul_count, 2)
        self.assertEqual(lifted_mul_count, 0)
