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


test_data_suite = [
    # (test_name, test_data)
    ("zeros", torch.zeros(1, 10, 10, 10)),
    ("ones", torch.ones(10, 10, 10)),
    ("rand", torch.rand(10, 10) - 0.5),
    ("randn_pos", torch.randn(10) + 10),
    ("randn_neg", torch.randn(10) - 10),
    ("ramp", torch.arange(-16, 16, 0.2)),
]


class TestHardsigmoid(unittest.TestCase):
    class Hardsigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hardsigmoid = torch.nn.Hardsigmoid()

        def forward(self, x):
            return self.hardsigmoid(x)

    def _test_hardsigmoid_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check(["torch.ops.aten.hardsigmoid.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_hardsigmoid_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .check(["torch.ops.aten.hardsigmoid.default"])
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_hardsigmoid_tosa_ethos_BI_pipeline(
        self,
        compile_spec: list[CompileSpec],
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
            .check_count({"torch.ops.aten.hardsigmoid.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_clamp_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_hardsigmoid_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_hardsigmoid_tosa_MI_pipeline(self.Hardsigmoid(), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_hardsigmoid_tosa_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_hardsigmoid_tosa_BI_pipeline(self.Hardsigmoid(), (test_data,))

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_hardsigmoid_tosa_u55_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_hardsigmoid_tosa_ethos_BI_pipeline(
            common.get_u55_compile_spec(), self.Hardsigmoid(), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_hardsigmoid_tosa_u85_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_hardsigmoid_tosa_ethos_BI_pipeline(
            common.get_u85_compile_spec(), self.Hardsigmoid(), (test_data,)
        )
