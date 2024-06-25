# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the unsqueeze op which copies the data of the input tensor (possibly with new data format)
#

import unittest
from typing import Sequence, Tuple

import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from executorch.backends.xnnpack.test.tester.tester import Quantize
from parameterized import parameterized


class TestSimpleUnsqueeze(unittest.TestCase):
    class Unsqueeze(torch.nn.Module):
        shapes: list[int | Sequence[int]] = [5, (5, 5), (5, 5), (5, 5, 5)]
        test_parameters: list[tuple[torch.Tensor]] = [(torch.ones(n),) for n in shapes]

        def forward(self, x: torch.Tensor, dim):
            return x.unsqueeze(dim)

    def _test_unsqueeze_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor, int]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check_count({"torch.ops.aten.unsqueeze.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_unsqueeze_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor, int]
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
            .check_count({"torch.ops.aten.unsqueeze.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_unsqueeze_tosa_u55_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor, int]
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
            .check_count({"torch.ops.aten.unsqueeze.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Unsqueeze.test_parameters)
    def test_unsqueeze_tosa_MI(self, test_tensor: torch.Tensor):
        for i in range(-test_tensor.dim() - 1, test_tensor.dim() + 1):
            self._test_unsqueeze_tosa_MI_pipeline(self.Unsqueeze(), (test_tensor, i))

    @parameterized.expand(Unsqueeze.test_parameters)
    def test_unsqueeze_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_unsqueeze_tosa_BI_pipeline(self.Unsqueeze(), (test_tensor, 0))

    @parameterized.expand(Unsqueeze.test_parameters)
    def test_unsqueeze_u55_BI(self, test_tensor: torch.Tensor):
        self._test_unsqueeze_tosa_u55_pipeline(self.Unsqueeze(), (test_tensor, 0))
