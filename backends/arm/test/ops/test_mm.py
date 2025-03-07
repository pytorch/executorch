# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from typing import Callable, Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestMM(unittest.TestCase):
    """Tests MatMul"""

    class MM(torch.nn.Module):
        test_data_generators = [
            lambda: (torch.rand(3, 5), torch.rand(5, 2)),
            lambda: (torch.rand(1, 1), torch.rand(1, 1)),
            lambda: (torch.ones(55, 3), torch.ones(3, 44)),
            lambda: (10000 * torch.randn(1, 10), torch.randn(10, 5)),
            lambda: (-10 * torch.randn(32, 64), 5 + 5 * torch.randn(64, 32)),
        ]

        def forward(self, x, y):
            return torch.mm(x, y)

    class MMSingleInput(torch.nn.Module):
        test_data_generators = [
            lambda: (torch.rand(3, 3),),
            lambda: (torch.ones(128, 128),),
            lambda: (10000 * torch.randn(25, 25),),
            lambda: (5 + 5 * torch.randn(64, 64),),
        ]

        def forward(self, x):
            return torch.mm(x, x)

    def _test_mm_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.mm.default": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_mm_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_mm_tosa_BI_pipeline(
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
            .check_count({"torch.ops.aten.mm.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_mm_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_mm_ethosu_BI_pipeline(
        self,
        compile_spec: CompileSpec,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.mm.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(MM.test_data_generators)
    def test_mm_tosa_MI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_mm_tosa_MI_pipeline(self.MM(), test_data)

    @parameterized.expand(MMSingleInput.test_data_generators)
    @pytest.mark.flaky  # TODO: Investigate flakyness (MLETORCH-534)
    def test_mm_single_input_tosa_MI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_mm_tosa_MI_pipeline(self.MMSingleInput(), test_data)

    @parameterized.expand(MM.test_data_generators)
    @pytest.mark.flaky  # TODO: Investigate flakyness (MLETORCH-534)
    def test_mm_tosa_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_mm_tosa_BI_pipeline(self.MM(), test_data)

    @parameterized.expand(MMSingleInput.test_data_generators)
    @pytest.mark.flaky  # TODO: Investigate flakyness (MLETORCH-534)
    def test_mm_single_input_tosa_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_mm_tosa_BI_pipeline(self.MMSingleInput(), test_data)

    # TODO: Enable numerical testing
    @parameterized.expand(MM.test_data_generators)
    def test_mm_u55_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_mm_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.MM(), test_data
        )

    # TODO: Enable numerical testing
    @parameterized.expand(MMSingleInput.test_data_generators)
    def test_mm_single_input_u55_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_mm_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.MMSingleInput(), test_data
        )

    @parameterized.expand(MM.test_data_generators)
    def test_mm_u85_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_mm_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.MM(), test_data
        )

    @parameterized.expand(MMSingleInput.test_data_generators)
    def test_mm_single_input_u85_BI(self, test_data_generator: Callable[[], Tuple]):
        test_data = test_data_generator()
        self._test_mm_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.MMSingleInput(), test_data
        )
