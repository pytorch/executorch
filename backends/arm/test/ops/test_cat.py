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


class TestCat(unittest.TestCase):

    class Cat(torch.nn.Module):
        test_parameters = [
            ((torch.ones(1), torch.ones(1)), 0),
            ((torch.ones(1, 2), torch.randn(1, 5), torch.randn(1, 1)), 1),
            (
                (
                    torch.ones(1, 2, 5),
                    torch.randn(1, 2, 4),
                    torch.randn(1, 2, 2),
                    torch.randn(1, 2, 1),
                ),
                -1,
            ),
            ((torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 1)), 3),
            (
                (
                    10000 * torch.randn(2, 3, 1, 4),
                    torch.randn(2, 7, 1, 4),
                    torch.randn(2, 1, 1, 4),
                ),
                -3,
            ),
        ]

        def __init__(self):
            super().__init__()

        def forward(self, tensors: tuple[torch.Tensor, ...], dim: int) -> torch.Tensor:
            return torch.cat(tensors, dim=dim)

    def _test_cat_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[tuple[torch.Tensor, ...], int]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check_count({"torch.ops.aten.cat.default": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_cat_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_cat_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[tuple[torch.Tensor, ...], int]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.cat.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_cat_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_cat_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[tuple[torch.Tensor, ...], int]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.cat.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_cat_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Cat.test_parameters)
    def test_cat_tosa_MI(self, operands: tuple[torch.Tensor, ...], dim: int):
        test_data = (operands, dim)
        self._test_cat_tosa_MI_pipeline(self.Cat(), test_data)

    def test_cat_4d_tosa_MI(self):
        square = torch.ones((2, 2, 2, 2))
        for dim in range(-3, 3):
            test_data = ((square, square), dim)
            self._test_cat_tosa_MI_pipeline(self.Cat(), test_data)

    @parameterized.expand(Cat.test_parameters)
    def test_cat_tosa_BI(self, operands: tuple[torch.Tensor, ...], dim: int):
        test_data = (operands, dim)
        self._test_cat_tosa_BI_pipeline(self.Cat(), test_data)

    # TODO: Remove @unittest.expectedFailure when this issue is fixed in Regor
    @parameterized.expand(Cat.test_parameters)
    def test_cat_u55_BI(self, operands: tuple[torch.Tensor, ...], dim: int):
        test_data = (operands, dim)
        self._test_cat_u55_BI_pipeline(self.Cat(), test_data)
