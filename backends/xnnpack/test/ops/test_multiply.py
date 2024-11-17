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
        def forward(self, x, y):
            z = x * y
            return z

    class Mul2(torch.nn.Module):
        def forward(self, x):
            z = x * x
            return z

    class MulFunctional(torch.nn.Module):
        def forward(self, x, y):
            z = torch.mul(x, y) * torch.functional.torch.mul(x, y)
            return z

    class MulRelu(torch.nn.Module):
        def forward(self, x, y):
            z = x * y
            return torch.nn.functional.relu(z)

    def _test_mul(self, inputs):
        for legacy in (True, False):
            tester = Tester(self.Mul(), inputs)
            tester.export()
            tester.check_count({"torch.ops.aten.mul.Tensor": 1})
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(["executorch_exir_dialects_edge__ops_aten_mul_Tensor"])
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_fp16_mul(self):
        inputs = (
            torch.randn((1, 3)).to(torch.float16),
            torch.randn((4, 3)).to(torch.float16),
        )
        self._test_mul(inputs)

    def test_fp32_mul(self):
        inputs = (torch.randn((1, 3)), torch.randn((4, 3)))
        self._test_mul(inputs)

    def test_qs8_mul(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1))
        for legacy in (True, False):
            tester = Tester(self.Mul(), inputs)
            tester.quantize()
            tester.export()
            tester.check_count({"torch.ops.aten.mul.Tensor": 1})
            tester.check(["torch.ops.quantized_decomposed"])
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_qs8_mul2(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        for legacy in (True, False):
            tester = Tester(self.Mul2(), inputs)
            tester.quantize()
            tester.export()
            tester.check_count({"torch.ops.aten.mul.Tensor": 1})
            tester.check(["torch.ops.quantized_decomposed"])
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_qs8_mul_functional(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        for legacy in (True, False):
            tester = Tester(self.MulFunctional(), inputs)
            tester.quantize()
            tester.export()
            tester.check_count({"torch.ops.aten.mul.Tensor": 3})
            tester.check(["torch.ops.quantized_decomposed"])
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()

    def test_qs8_mul_relu(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        for legacy in (True, False):
            tester = Tester(self.MulRelu(), inputs)
            tester.quantize()
            tester.export()
            tester.check_count(
                {
                    "torch.ops.aten.mul.Tensor": 1,
                    "torch.ops.aten.relu.default": 1,
                }
            )
            tester.check(["torch.ops.quantized_decomposed"])
            if legacy:
                tester.to_edge()
                tester.partition()
            else:
                tester.to_edge_transform_and_lower()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "executorch_exir_dialects_edge__ops_aten_relu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            tester.to_executorch()
            tester.serialize()
            tester.run_method_and_compare_outputs()
