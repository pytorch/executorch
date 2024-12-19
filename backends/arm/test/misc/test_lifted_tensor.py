# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from typing import Union

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized


class LiftedTensor(torch.nn.Module):

    test_data = [
        # (operator, test_data, length)
        (operator.add, (torch.randn(2, 2), 2)),
        (operator.truediv, (torch.ones(2, 2), 2)),
        (operator.mul, (torch.randn(2, 2), 2)),
        (operator.sub, (torch.rand(2, 2), 2)),
    ]

    def __init__(self, op: callable):
        super().__init__()
        self.op = op
        self.lifted_tensor = torch.Tensor([[1, 2], [3, 4]])

    def forward(self, x: torch.Tensor, length) -> torch.Tensor:
        sliced = self.lifted_tensor[:, :length]
        return self.op(sliced, x)


class LiftedScalarTensor(torch.nn.Module):
    test_data = [
        # (operator, test_data)
        (operator.add, (torch.randn(2, 2),), 1.0),
        (operator.truediv, (torch.randn(4, 2),), 1.0),
        (operator.mul, (torch.randn(1, 2),), 2.0),
        (operator.sub, (torch.randn(3),), 1.0),
    ]

    def __init__(self, op: callable, arg1: Union[int, float, torch.tensor]):
        super().__init__()
        self.op = op
        self.arg1 = arg1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x, self.arg1)


class TestLiftedTensor(unittest.TestCase):
    """Tests the ArmPartitioner with a placeholder of type lifted tensor."""

    @parameterized.expand(LiftedTensor.test_data)
    def test_partition_lifted_tensor_tosa_MI(self, op, data):
        tester = (
            ArmTester(
                LiftedTensor(op),
                example_inputs=data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .to_edge()
        )
        signature = tester.get_artifact().exported_program().graph_signature
        assert len(signature.lifted_tensor_constants) > 0
        tester.partition()
        tester.to_executorch()
        tester.run_method_and_compare_outputs(data)

    @parameterized.expand(LiftedTensor.test_data)
    def test_partition_lifted_tensor_tosa_BI(self, op, data):
        tester = (
            ArmTester(
                LiftedTensor(op),
                example_inputs=data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .to_edge()
        )
        signature = tester.get_artifact().exported_program().graph_signature
        assert len(signature.lifted_tensor_constants) == 0
        tester.partition()
        tester.to_executorch()
        tester.run_method_and_compare_outputs(data)

    @parameterized.expand(LiftedScalarTensor.test_data)
    def test_partition_lifted_scalar_tensor_tosa_MI(self, op, data, arg1):
        (
            ArmTester(
                LiftedScalarTensor(op, arg1),
                example_inputs=(data),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .run_method_and_compare_outputs(data)
        )

    @parameterized.expand(LiftedScalarTensor.test_data)
    def test_partition_lifted_scalar_tensor_tosa_BI(self, op, data, arg1):
        (
            ArmTester(
                LiftedScalarTensor(op, arg1),
                example_inputs=(data),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .run_method_and_compare_outputs(data)
        )
