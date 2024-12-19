# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests the rsqrt op.
#

import unittest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


class TestRsqrt(unittest.TestCase):
    class Rsqrt(torch.nn.Module):
        test_parameters = [
            (torch.ones(1, 10, 10, 10),),
            (torch.rand(1, 10, 10, 10),),
            (torch.rand(1, 5, 10, 20),),
            (torch.rand(5, 10, 20),),
        ]

        def forward(self, x: torch.Tensor):
            return x.rsqrt()

    def _test_rsqrt_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.rsqrt.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_rsqrt_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.rsqrt.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_rsqrt_ethosu_BI_pipeline(
        self,
        compile_spec: CompileSpec,
        module: torch.nn.Module,
        test_data: tuple[torch.Tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.rsqrt.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Rsqrt.test_parameters)
    def test_rsqrt_tosa_MI(self, test_tensor: torch.Tensor):
        self._test_rsqrt_tosa_MI_pipeline(self.Rsqrt(), (test_tensor,))

    @parameterized.expand(Rsqrt.test_parameters)
    def test_rsqrt_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_rsqrt_tosa_BI_pipeline(self.Rsqrt(), (test_tensor,))

    @parameterized.expand(Rsqrt.test_parameters)
    def test_rsqrt_u55_BI(self, test_tensor: torch.Tensor):
        self._test_rsqrt_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Rsqrt(), (test_tensor,)
        )

    @parameterized.expand(Rsqrt.test_parameters)
    def test_rsqrt_u85_BI(self, test_tensor: torch.Tensor):
        self._test_rsqrt_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Rsqrt(), (test_tensor,)
        )
