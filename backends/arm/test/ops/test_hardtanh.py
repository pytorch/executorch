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

from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.xnnpack.test.tester.tester import Quantize
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


class TestHardTanh(unittest.TestCase):
    """Tests HardTanh Operator."""

    class HardTanh(torch.nn.Module):

        def __init__(self):
            super().__init__()

            self.hardTanh = torch.nn.Hardtanh()

        def forward(self, x):
            return self.hardTanh(x)

    def _test_hardtanh_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check(["torch.ops.aten.hardtanh.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_hardtanh_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_hardtanh_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.hardtanh.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_hardtanh_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_hardtanh_tosa_ethosu_BI_pipeline(
        self, compile_spec, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.hardtanh.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_hardtanh_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_hardtanh_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_hardtanh_tosa_MI_pipeline(self.HardTanh(), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_hardtanh_tosa_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_hardtanh_tosa_BI_pipeline(self.HardTanh(), (test_data,))

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_hardtanh_tosa_u55_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_hardtanh_tosa_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.HardTanh(), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_hardtanh_tosa_u85_BI(self, test_name: str, test_data: torch.Tensor):
        self._test_hardtanh_tosa_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.HardTanh(), (test_data,)
        )
