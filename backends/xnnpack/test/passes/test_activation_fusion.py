# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack._passes.fuse_activation_pass import FuseActivationPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops


class TestActivationFusion(unittest.TestCase):
    PassStage = RunPasses([ConvertToLinearPass, FuseActivationPass])

    def check_node_has_tag(self, graph_module, node_target, tag):
        for n in graph_module.graph.nodes:
            if n.op == "call_function" and n.target == node_target:
                return FuseActivationPass.FUSED_ACTIVATION_TAG in n.meta

    class OpActivation(torch.nn.Module):
        def __init__(self, module: torch.nn.Module, activation):
            super().__init__()
            self.seq = torch.nn.Sequential(module, activation)

        def forward(self, x):
            return self.seq(x)

    class UnaryOps(torch.nn.Module):
        def __init__(self, unary_op):
            super().__init__()
            self.unary_op = unary_op

        def forward(self, a):
            return self.unary_op(a, a)

    def _test_op_activation_case(
        self,
        module,
        edge_op,
        inputs,
        quantize=False,
        activation=None,
        activation_name="executorch_exir_dialects_edge__ops_aten_relu_default",
    ):
        activation = activation or torch.nn.ReLU()
        tester = Tester(self.OpActivation(module, activation).eval(), inputs)
        if quantize:
            tester.quantize()

        artifact = (
            tester.export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_not([activation_name])
            .get_artifact(Tester.stage_name(self.PassStage))
        )

        for node in artifact.exported_program().module().graph.nodes:
            if node.op == "call_function" and node.target == edge_op:
                self.assertTrue(FuseActivationPass.FUSED_ACTIVATION_TAG in node.meta)

    def test_activation_fusion_conv_relu(self):
        inputs = (torch.randn(1, 1, 8, 8),)
        self._test_op_activation_case(
            torch.nn.Conv2d(1, 1, (4, 4)),
            exir_ops.edge.aten.convolution.default,
            inputs,
        )
        self._test_op_activation_case(
            torch.nn.Conv2d(1, 1, (4, 4)),
            exir_ops.edge.aten.convolution.default,
            inputs,
            quantize=True,
        )

    def test_activation_fusion_linear_relu(self):
        inputs = (torch.randn(1, 1, 8, 8),)
        self._test_op_activation_case(
            torch.nn.Linear(8, 8),
            exir_ops.edge.aten.linear.default,
            inputs,
        )
        self._test_op_activation_case(
            torch.nn.Linear(8, 8),
            exir_ops.edge.aten.linear.default,
            inputs,
            quantize=True,
        )

    def test_activation_fusion_add_relu(self):
        inputs = (torch.randn(1, 1, 8, 8),)

        self._test_op_activation_case(
            self.UnaryOps(torch.add),
            exir_ops.edge.aten.add.Tensor,
            inputs,
        )
        self._test_op_activation_case(
            self.UnaryOps(torch.add),
            exir_ops.edge.aten.add.Tensor,
            inputs,
            quantize=True,
        )

    def test_activation_fusion_mul_relu(self):
        inputs = (torch.randn(1, 1, 8, 8),)

        self._test_op_activation_case(
            self.UnaryOps(torch.mul),
            exir_ops.edge.aten.mul.Tensor,
            inputs,
        )
        self._test_op_activation_case(
            self.UnaryOps(torch.mul),
            exir_ops.edge.aten.mul.Tensor,
            inputs,
            quantize=True,
        )

    def test_activation_fusion_sub_relu(self):
        inputs = (torch.randn(1, 1, 8, 8),)

        self._test_op_activation_case(
            self.UnaryOps(torch.sub),
            exir_ops.edge.aten.sub.Tensor,
            inputs,
        )
        self._test_op_activation_case(
            self.UnaryOps(torch.sub),
            exir_ops.edge.aten.sub.Tensor,
            inputs,
            quantize=True,
        )

    def test_activation_fusion_conv_hardtanh(self):
        inputs = (torch.randn(1, 1, 8, 8),)
        self._test_op_activation_case(
            torch.nn.Conv2d(1, 1, (4, 4)),
            exir_ops.edge.aten.convolution.default,
            inputs,
            activation=torch.nn.Hardtanh(min_val=-1.0, max_val=1.0),
            activation_name="executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        )
        self._test_op_activation_case(
            torch.nn.Conv2d(1, 1, (4, 4)),
            exir_ops.edge.aten.convolution.default,
            inputs,
            activation=torch.nn.Hardtanh(min_val=-1.0, max_val=1.0),
            activation_name="executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        )

    def test_activation_fusion_linear_hardtanh(self):
        inputs = (torch.randn(1, 1, 8, 8),)
        self._test_op_activation_case(
            torch.nn.Linear(8, 8),
            exir_ops.edge.aten.linear.default,
            inputs,
            activation=torch.nn.Hardtanh(min_val=-1.0, max_val=1.0),
            activation_name="executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        )

    def test_activation_fusion_add_hardtanh(self):
        inputs = (torch.randn(1, 1, 8, 8),)

        self._test_op_activation_case(
            self.UnaryOps(torch.add),
            exir_ops.edge.aten.add.Tensor,
            inputs,
            activation=torch.nn.Hardtanh(min_val=-1.0, max_val=1.0),
            activation_name="executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        )

    def test_activation_fusion_mul_hardtanh(self):
        inputs = (torch.randn(1, 1, 8, 8),)

        self._test_op_activation_case(
            self.UnaryOps(torch.mul),
            exir_ops.edge.aten.mul.Tensor,
            inputs,
            activation=torch.nn.Hardtanh(min_val=-1.0, max_val=1.0),
            activation_name="executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        )

    def test_activation_fusion_sub_hardtanh(self):
        inputs = (torch.randn(1, 1, 8, 8),)

        self._test_op_activation_case(
            self.UnaryOps(torch.sub),
            exir_ops.edge.aten.sub.Tensor,
            inputs,
            activation=torch.nn.Hardtanh(min_val=-1.0, max_val=1.0),
            activation_name="executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        )
