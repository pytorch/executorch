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
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


class TestMinimum(unittest.TestCase):
    """Tests a single minimum op"""

    class Minimum(torch.nn.Module):
        test_parameters = [
            (
                torch.FloatTensor([1, 2, 3, 5, 7]),
                (torch.FloatTensor([2, 1, 2, 1, 10])),
            ),
            (torch.ones(1, 10, 4, 6), 2 * torch.ones(1, 10, 4, 6)),
            (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
            (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)),
            (10000 * torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1)),
        ]

        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.minimum(x, y)

    def _test_minimum_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.minimum.default": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_minimum_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.minimum.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_minimum_ethos_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.Tensor],
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .to_executorch()
            .serialize()
        )

        return tester

    @parameterized.expand(Minimum.test_parameters)
    def test_minimum_tosa_MI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_minimum_tosa_MI_pipeline(self.Minimum(), test_data)

    @parameterized.expand(Minimum.test_parameters)
    def test_minimum_tosa_BI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        self._test_minimum_tosa_BI_pipeline(self.Minimum(), test_data)

    @parameterized.expand(Minimum.test_parameters)
    @unittest.expectedFailure  # Bug in Vela, disabled until pin changes, bug MLETORCH-513
    def test_minimum_u55_BI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        tester = self._test_minimum_ethos_BI_pipeline(
            self.Minimum(), common.get_u55_compile_spec(), test_data
        )
        if common.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                qtol=1, inputs=test_data, target_board="corstone-300"
            )

    @parameterized.expand(Minimum.test_parameters)
    def test_minimum_u85_BI(self, operand1: torch.Tensor, operand2: torch.Tensor):
        test_data = (operand1, operand2)
        tester = self._test_minimum_ethos_BI_pipeline(
            self.Minimum(), common.get_u85_compile_spec(), test_data
        )
        if common.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(
                qtol=1, inputs=test_data, target_board="corstone-320"
            )
