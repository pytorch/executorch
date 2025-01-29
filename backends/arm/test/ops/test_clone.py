# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the clone op which copies the data of the input tensor (possibly with new data format)
#

import unittest
from typing import Tuple

import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.tosa_specification import TosaSpecification

from executorch.backends.xnnpack.test.tester.tester import Quantize

from parameterized import parameterized


class TestSimpleClone(unittest.TestCase):
    """Tests clone."""

    class Clone(torch.nn.Module):
        sizes = [10, 15, 50, 100]
        test_parameters = [(torch.ones(n),) for n in sizes]

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            x = x.clone()
            return x

    def _test_clone_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: torch.Tensor
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.clone.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_clone_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        tosa_spec = TosaSpecification.create_from_string("TOSA-0.80+BI")
        compile_spec = common.get_tosa_compile_spec(tosa_spec)
        quantizer = ArmQuantizer(tosa_spec).set_io(get_symmetric_quantization_config())
        (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.clone.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    @parameterized.expand(Clone.test_parameters)
    def test_clone_tosa_MI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_MI_pipeline(self.Clone(), (test_tensor,))

    @parameterized.expand(Clone.test_parameters)
    def test_clone_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_BI_pipeline(self.Clone(), (test_tensor,))
