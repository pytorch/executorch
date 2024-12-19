# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the repeat op which copies the data of the input tensor (possibly with new data format)
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
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized


class TestSimpleRepeat(unittest.TestCase):
    """Tests Tensor.repeat for different ranks and dimensions."""

    class Repeat(torch.nn.Module):
        # (input tensor, multiples)
        test_parameters = [
            (torch.randn(3), (2,)),
            (torch.randn(3, 4), (2, 1)),
            (torch.randn(1, 1, 2, 2), (1, 2, 3, 4)),
            (torch.randn(3), (2, 2)),
            (torch.randn(3), (1, 2, 3)),
            (torch.randn((3, 3)), (2, 2, 2)),
            (torch.randn((3, 3, 3)), (2, 1, 2, 4)),
        ]

        def forward(self, x: torch.Tensor, multiples: Sequence):
            return x.repeat(multiples)

    def _test_repeat_tosa_MI_pipeline(self, module: torch.nn.Module, test_data: Tuple):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.repeat.default": 1})
            .to_edge()
            .partition()
            .check_not(["torch.ops.aten.repeat.default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_repeat_tosa_BI_pipeline(self, module: torch.nn.Module, test_data: Tuple):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.repeat.default": 1})
            .to_edge()
            .partition()
            .check_not(["torch.ops.aten.repeat.default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_repeat_ethosu_pipeline(
        self, compile_spec: CompileSpec, module: torch.nn.Module, test_data: Tuple
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.repeat.default": 1})
            .to_edge()
            .partition()
            .check_not(["torch.ops.aten.repeat.default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Repeat.test_parameters)
    def test_repeat_tosa_MI(self, test_input, multiples):
        self._test_repeat_tosa_MI_pipeline(self.Repeat(), (test_input, multiples))

    @parameterized.expand(Repeat.test_parameters)
    def test_repeat_tosa_BI(self, test_input, multiples):
        self._test_repeat_tosa_BI_pipeline(self.Repeat(), (test_input, multiples))

    @parameterized.expand(Repeat.test_parameters[:-1])
    def test_repeat_u55_BI(self, test_input, multiples):
        self._test_repeat_ethosu_pipeline(
            common.get_u55_compile_spec(), self.Repeat(), (test_input, multiples)
        )

    # Final test requires transpose which is not supported on u55.
    @parameterized.expand(Repeat.test_parameters[-1:])
    @unittest.expectedFailure
    def test_repeat_u55_BI_xfails(self, test_input, multiples):
        self._test_repeat_ethosu_pipeline(
            common.get_u55_compile_spec(), self.Repeat(), (test_input, multiples)
        )

    @parameterized.expand(Repeat.test_parameters)
    def test_repeat_u85_BI(self, test_input, multiples):
        self._test_repeat_ethosu_pipeline(
            common.get_u85_compile_spec(), self.Repeat(), (test_input, multiples)
        )
