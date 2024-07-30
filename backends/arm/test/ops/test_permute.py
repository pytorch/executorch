# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.xnnpack.test.tester.tester import Quantize
from parameterized import parameterized
from torchvision.ops import Permute

test_data_suite = [
    # (test_name,test_data,dims)
    ("zeros", torch.zeros(10, 10, 10, 10), [1, 0, 3, 2]),
    ("ones", torch.ones(10, 10, 10, 10), [3, 1, 0, 2]),
    ("rand", torch.rand(10, 10, 10, 10) - 0.5, [0, 2, 3, 1]),
    ("randn_pos", torch.randn(10, 10, 10) + 10, [2, 0, 1]),
    ("randn_neg", torch.randn(10, 10, 10) - 10, [1, 2, 0]),
    ("ramp", torch.arange(-16, 16, 0.2), [0]),
]


class TestPermute(unittest.TestCase):
    """Tests Permute Operator."""

    class Permute(torch.nn.Module):

        def __init__(self, dims: list[int]):
            super().__init__()

            self.permute = Permute(dims=dims)

        def forward(self, x):
            return self.permute(x)

    def _test_permute_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check(["torch.ops.aten.permute.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_permute_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.permute.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_permute_tosa_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.permute.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(test_data_suite)
    def test_permute_tosa_MI(
        self, test_name: str, test_data: torch.Tensor, dims: list[int]
    ):
        self._test_permute_tosa_MI_pipeline(self.Permute(dims=dims), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_permute_tosa_BI(
        self, test_name: str, test_data: torch.Tensor, dims: list[int]
    ):
        self._test_permute_tosa_BI_pipeline(self.Permute(dims=dims), (test_data,))

    # Expected to fail as Permute is not supported by the NPU
    @parameterized.expand(test_data_suite)
    @unittest.expectedFailure
    def test_permute_tosa_u55_BI(
        self, test_name: str, test_data: torch.Tensor, dims: list[int]
    ):
        self._test_permute_tosa_u55_BI_pipeline(self.Permute(dims=dims), (test_data,))
