# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized

test_data_sute = [
    # (test_name, input, other,) See torch.mul() for info
    (
        "op_mul_rank1_rand",
        torch.rand(5) * 3.7,
        torch.rand(5) * 1.5,
    ),
    (
        "op_mul_rank2_rand",
        torch.rand(4, 5),
        torch.rand(1, 5),
    ),
    (
        "op_mul_rank3_randn",
        torch.randn(10, 5, 2),
        torch.randn(10, 5, 2),
    ),
    (
        "op_mul_rank4_randn",
        torch.randn(1, 10, 25, 20),
        torch.randn(1, 10, 25, 20),
    ),
    (
        "op_mul_rank4_ones_mul_negative",
        torch.ones(1, 10, 25, 20),
        (-1) * torch.ones(1, 10, 25, 20),
    ),
    (
        "op_mul_rank4_negative_large_rand",
        (-200) * torch.rand(1, 10, 25, 20),
        torch.rand(1, 1, 1, 20),
    ),
    (
        "op_mul_rank4_large_randn",
        200 * torch.randn(1, 10, 25, 20),
        torch.rand(1, 10, 25, 1),
    ),
]


class TestMul(unittest.TestCase):
    class Mul(torch.nn.Module):

        def forward(
            self,
            input_: torch.Tensor,
            other_: torch.Tensor,
        ):
            return input_ * other_

    def _test_mul_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[torch.Tensor, torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+MI", permute_memory_to_nhwc=True
                ),
            )
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_mul_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[torch.Tensor, torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+BI", permute_memory_to_nhwc=True
                ),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1.0)
        )

    def _test_mul_ethosu_BI_pipeline(
        self,
        compile_spec: CompileSpec,
        module: torch.nn.Module,
        test_data: tuple[torch.Tensor, torch.Tensor],
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_sute)
    def test_mul_tosa_MI(
        self,
        test_name: str,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        test_data = (input_, other_)
        self._test_mul_tosa_MI_pipeline(self.Mul(), test_data)

    @parameterized.expand(test_data_sute)
    def test_mul_tosa_BI(
        self,
        test_name: str,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):

        test_data = (input_, other_)
        self._test_mul_tosa_BI_pipeline(self.Mul(), test_data)

    @parameterized.expand(test_data_sute)
    @pytest.mark.corstone_fvp
    def test_mul_u55_BI(
        self,
        test_name: str,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        test_data = (input_, other_)
        self._test_mul_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Mul(), test_data
        )

    @parameterized.expand(test_data_sute)
    @pytest.mark.corstone_fvp
    def test_mul_u85_BI(
        self,
        test_name: str,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        test_data = (input_, other_)
        self._test_mul_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Mul(), test_data
        )
