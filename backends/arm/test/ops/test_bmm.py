# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Callable, Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


class TestBMM(unittest.TestCase):
    """Tests Batch MatMul"""

    class BMM(torch.nn.Module):
        test_data_generators = [
            lambda: (torch.rand(2, 1, 1), torch.rand(2, 1, 1)),
            lambda: (torch.rand(5, 3, 5), torch.rand(5, 5, 2)),
            lambda: (torch.ones(1, 55, 3), torch.ones(1, 3, 44)),
            lambda: (10000 * torch.randn(10, 1, 10), torch.randn(10, 10, 5)),
            lambda: (-10 * torch.randn(2, 32, 64), 5 + 5 * torch.randn(2, 64, 32)),
        ]

        def forward(self, x, y):
            return torch.bmm(x, y)

    class MatMul(torch.nn.Module):
        test_data_generators = [
            lambda: (torch.rand(2, 3, 5), torch.rand(2, 5, 2)),
            lambda: (torch.rand(1, 2, 3, 5), torch.rand(1, 2, 5, 2)),
        ]

        def forward(self, x, y):
            return torch.matmul(x, y)

    class BMMSingleInput(torch.nn.Module):
        test_data_generators = [
            lambda: (torch.rand(20, 3, 3),),
            lambda: (torch.rand(2, 128, 128),),
            lambda: (10000 * torch.randn(4, 25, 25),),
            lambda: (5 + 5 * torch.randn(3, 64, 64),),
        ]

        def forward(self, x):
            return torch.bmm(x, x)

    def _test_bmm_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor, ...]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_bmm_default": 1})
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_bmm_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_bmm_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor, ...]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_bmm_default": 1})
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_bmm_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_bmm_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.Tensor, ...],
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.bmm.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(inputs=test_data, qtol=1)

    @parameterized.expand(BMM.test_data_generators)
    def test_bmm_tosa_MI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_tosa_MI_pipeline(self.BMM(), test_data)

    @parameterized.expand(BMMSingleInput.test_data_generators)
    def test_bmm_single_input_tosa_MI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_tosa_MI_pipeline(self.BMMSingleInput(), test_data)

    @parameterized.expand(MatMul.test_data_generators)
    def test_matmul_tosa_MI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_tosa_MI_pipeline(self.MatMul(), test_data)

    @parameterized.expand(MatMul.test_data_generators)
    def test_matmul_tosa_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_tosa_BI_pipeline(self.MatMul(), test_data)

    @parameterized.expand(BMM.test_data_generators)
    def test_bmm_tosa_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_tosa_BI_pipeline(self.BMM(), test_data)

    @parameterized.expand(BMMSingleInput.test_data_generators)
    def test_bmm_single_input_tosa_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_tosa_BI_pipeline(self.BMMSingleInput(), test_data)

    @parameterized.expand(BMM.test_data_generators)
    @pytest.mark.corstone_fvp
    @unittest.expectedFailure
    def test_bmm_u55_BI_xfails(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_ethosu_BI_pipeline(
            self.BMM(), common.get_u55_compile_spec(), test_data
        )

    @parameterized.expand(BMM.test_data_generators)
    @pytest.mark.corstone_fvp
    def test_bmm_u85_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_ethosu_BI_pipeline(
            self.BMM(), common.get_u85_compile_spec(), test_data
        )

    # Expected to fail with error: Warning, unsupported fusing of TOSA Rescale previous operator is of type: Memcpy
    @parameterized.expand(BMMSingleInput.test_data_generators)
    @pytest.mark.corstone_fvp
    @unittest.expectedFailure
    def test_bmm_single_input_u55_BI_xfails(
        self, test_data_generator: Callable[[], Tuple]
    ):
        test_data = test_data_generator()
        self._test_bmm_ethosu_BI_pipeline(
            self.BMMSingleInput(), common.get_u55_compile_spec(), test_data
        )

    @parameterized.expand(BMMSingleInput.test_data_generators)
    @pytest.mark.corstone_fvp
    def test_bmm_single_input_u85_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_bmm_ethosu_BI_pipeline(
            self.BMMSingleInput(), common.get_u85_compile_spec(), test_data
        )
