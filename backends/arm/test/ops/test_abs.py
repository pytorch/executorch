# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


class TestAbs(unittest.TestCase):
    class Abs(torch.nn.Module):
        test_parameters = [
            (torch.zeros(5),),
            (torch.full((5,), -1, dtype=torch.float32),),
            (torch.ones(5) * -1,),
            (torch.randn(8),),
            (torch.randn(2, 3, 4),),
            (torch.randn(1, 2, 3, 4),),
            (torch.normal(mean=0, std=10, size=(2, 3, 4)),),
        ]

        def forward(self, x):
            return torch.abs(x)

    def _test_abs_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.abs.default": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["torch.ops.aten.abs.default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_abs_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.abs.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_abs_ethosu_BI_pipeline(
        self,
        compile_spec: list[CompileSpec],
        module: torch.nn.Module,
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
            .check_count({"torch.ops.aten.abs.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(Abs.test_parameters)
    def test_abs_tosa_MI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_abs_tosa_MI_pipeline(self.Abs(), test_data)

    @parameterized.expand(Abs.test_parameters)
    def test_abs_tosa_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_abs_tosa_BI_pipeline(self.Abs(), test_data)

    @parameterized.expand(Abs.test_parameters)
    @pytest.mark.corstone_fvp
    def test_abs_u55_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_abs_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Abs(), test_data
        )

    @parameterized.expand(Abs.test_parameters)
    @pytest.mark.corstone_fvp
    def test_abs_u85_BI(self, test_data: torch.Tensor):
        test_data = (test_data,)
        self._test_abs_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Abs(), test_data
        )
