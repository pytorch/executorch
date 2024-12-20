# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

test_data_t = tuple[str, torch.Tensor]
test_data_suite: list[test_data_t] = [
    (
        "op_reciprocal_rank1_ones",
        torch.ones(5),
    ),
    (
        "op_reciprocal_rank1_rand",
        torch.rand(5) * 5,
    ),
    ("op_reciprocal_rank1_negative_ones", torch.ones(5) * (-1)),
    ("op_reciprocal_rank4_ones", torch.ones(1, 10, 25, 20)),
    ("op_reciprocal_rank4_negative_ones", (-1) * torch.ones(1, 10, 25, 20)),
    ("op_reciprocal_rank4_ones_reciprocal_negative", torch.ones(1, 10, 25, 20)),
    ("op_reciprocal_rank4_large_rand", 200 * torch.rand(1, 10, 25, 20)),
    ("op_reciprocal_rank4_negative_large_rand", (-200) * torch.rand(1, 10, 25, 20)),
    ("op_reciprocal_rank4_large_randn", 200 * torch.randn(1, 10, 25, 20) + 1),
]


class TestReciprocal(unittest.TestCase):
    """Tests reciprocal"""

    class Reciprocal(torch.nn.Module):

        def forward(self, input_: torch.Tensor):
            return input_.reciprocal()

    def _test_reciprocal_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.reciprocal.default": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_reciprocal_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.reciprocal.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_reciprocal_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[torch.Tensor]
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.reciprocal.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_reciprocal_tosa_MI(self, test_name: str, input_: torch.Tensor):
        test_data = (input_,)
        self._test_reciprocal_tosa_MI_pipeline(self.Reciprocal(), test_data)

    @parameterized.expand(test_data_suite)
    def test_reciprocal_tosa_BI(self, test_name: str, input_: torch.Tensor):

        test_data = (input_,)
        self._test_reciprocal_tosa_BI_pipeline(self.Reciprocal(), test_data)

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_reciprocal_u55_BI(self, test_name: str, input_: torch.Tensor):
        test_data = (input_,)
        self._test_reciprocal_u55_BI_pipeline(self.Reciprocal(), test_data)
