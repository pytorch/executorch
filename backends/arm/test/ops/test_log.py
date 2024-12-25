# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
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
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized

test_data_suite = [
    # (test_name, test_data)
    ("ones_rank4", torch.ones(1, 10, 10, 10)),
    ("ones_rank3", torch.ones(10, 10, 10)),
    ("rand", torch.rand(10, 10) + 0.001),
    ("randn_pos", torch.randn(10) + 10),
    ("randn_spread", torch.max(torch.Tensor([0.0]), torch.randn(10) * 100)),
    ("ramp", torch.arange(0.01, 20, 0.2)),
]


class TestLog(unittest.TestCase):
    """Tests lowering of aten.log"""

    class Log(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.log(x)

    def _test_log_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check(["torch.ops.aten.log.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_log_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_log_tosa_BI_pipeline(self, module: torch.nn.Module, test_data: Tuple):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .check(["torch.ops.aten.log.default"])
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_log_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_log_ethosu_BI_pipeline(
        self,
        compile_spec: CompileSpec,
        module: torch.nn.Module,
        test_data: Tuple[torch.tensor],
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.log.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_log_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_log_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_log_tosa_MI_pipeline(self.Log(), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_log_tosa_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_log_tosa_BI_pipeline(self.Log(), (test_data,))

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_log_tosa_u55_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_log_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Log(), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_log_tosa_u85_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_log_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Log(), (test_data,)
        )
