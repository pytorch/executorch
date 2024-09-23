# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized


class TestSimpleSub(unittest.TestCase):
    class Sub(torch.nn.Module):
        test_parameters = [
            (torch.ones(5),),
            (3 * torch.ones(8),),
            (10 * torch.randn(8),),
        ]

        def forward(self, x):
            return x - x

    class Sub2(torch.nn.Module):
        test_parameters = [
            (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
        ]

        def forward(self, x, y):
            return x - y

    def _test_sub_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["torch.ops.aten.sub.Tensor"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_sub_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_sub_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Sub.test_parameters)
    def test_sub_tosa_MI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_sub_tosa_MI_pipeline(self.Sub(), test_data)

    @parameterized.expand(Sub.test_parameters)
    def test_sub_tosa_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_sub_tosa_BI_pipeline(self.Sub(), test_data)

    @parameterized.expand(Sub.test_parameters)
    def test_sub_u55_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_sub_u55_BI_pipeline(self.Sub(), test_data)

    @parameterized.expand(Sub2.test_parameters)
    def test_sub2_tosa_MI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_sub_tosa_MI_pipeline(self.Sub2(), test_data)
