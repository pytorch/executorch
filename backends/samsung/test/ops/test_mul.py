# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.


import unittest

import torch

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig


class Mul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class MulConstant(torch.nn.Module):
    def __init__(self, constant) -> None:
        super().__init__()
        self.constant = constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.constant


class TestMul(unittest.TestCase):
    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_mul_Tensor"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def _test_a8w8(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.quantize()
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_mul_Tensor"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs, atol=0.2)
        )

    def test_fp32_simple_mul(self):
        inputs = (torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8))
        self._test(Mul(), inputs)

    def test_fp32_const_mul(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test(MulConstant(torch.randn(1, 3, 8, 8)), inputs)

    def test_fp32_mul_broadcast(self):
        inputs = (torch.randn(1, 1, 8, 8), torch.randn(1, 3, 8, 8))
        self._test(Mul(), inputs)

    def test_a8w8_simple_mul(self):
        inputs = (torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8))
        self._test_a8w8(Mul(), inputs)

    def test_a8w8_const_mul(self):
        inputs = (torch.randn(1, 3, 8, 8),)
        self._test_a8w8(MulConstant(torch.randn(1, 3, 8, 8)), inputs)

    def test_a8w8_mul_broadcast(self):
        inputs = (torch.randn(1, 1, 8, 8), torch.randn(1, 3, 8, 8))
        self._test_a8w8(Mul(), inputs)
