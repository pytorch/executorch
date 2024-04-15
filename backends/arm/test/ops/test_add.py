# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from typing import Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestSimpleAdd(unittest.TestCase):
    class Add(torch.nn.Module):
        test_parameters = [
            (torch.ones(5),),
            (3 * torch.ones(8),),
            (10 * torch.randn(8),),
        ]

        def __init__(self):
            super().__init__()
            self.permute_memory_to_nhwc = False

        def forward(self, x):
            return x + x

    class Add2(torch.nn.Module):
        test_parameters = [
            (torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4)),
            (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
            (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
            (10000 * torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
        ]

        def __init__(self):
            super().__init__()
            self.permute_memory_to_nhwc = False

        def forward(self, x, y):
            return x + y

    def _test_add_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        tester = (
            ArmTester(
                module,
                inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )
        if common.TOSA_REF_MODEL_INSTALLED:
            tester.run_method_and_compare_outputs()
        else:
            logger.warning(
                "TOSA ref model tool not installed, skip numerical correctness tests"
            )

    def _test_add_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        tester = (
            ArmTester(
                module,
                inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

        if common.TOSA_REF_MODEL_INSTALLED:
            tester.run_method_and_compare_outputs(qtol=1)
        else:
            logger.warning(
                "TOSA ref model tool not installed, skip numerical correctness tests"
            )

    def _test_add_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Add.test_parameters)
    def test_add_tosa_MI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_add_tosa_MI_pipeline(self.Add(), test_data)

    @parameterized.expand(Add.test_parameters)
    def test_add_tosa_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_add_tosa_BI_pipeline(self.Add(), test_data)

    @parameterized.expand(Add.test_parameters)
    @unittest.skipIf(
        not common.VELA_INSTALLED,
        "There is no point in running U55 tests if the Vela tool is not installed",
    )
    def test_add_u55_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_add_u55_BI_pipeline(self.Add(), test_data)

    @parameterized.expand(Add2.test_parameters)
    def test_add2_tosa_MI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_add_tosa_MI_pipeline(self.Add2(), test_data)

    @parameterized.expand(Add2.test_parameters)
    def test_add2_tosa_BI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_add_tosa_BI_pipeline(self.Add2(), test_data)

    @parameterized.expand(Add2.test_parameters)
    @unittest.skipIf(
        not common.VELA_INSTALLED,
        "There is no point in running U55 tests if the Vela tool is not installed",
    )
    def test_add2_u55_BI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_add_u55_BI_pipeline(self.Add2(), test_data)
