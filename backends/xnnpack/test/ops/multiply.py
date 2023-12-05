# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestMul(unittest.TestCase):
    class Mul(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x * y
            return z

    def test_fp32_mul(self):
        inputs = (torch.randn((1, 3)), torch.randn((4, 3)))
        (
            Tester(self.Mul(), inputs)
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mul_Tensor"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )

    def test_qs8_mul(self):
        inputs = (torch.randn((1, 3)), torch.randn((4, 3)))
        (
            Tester(self.Mul(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
