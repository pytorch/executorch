# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from executorch.backends.xnnpack.utils.configs import (
    get_transform_passes,
    get_xnnpack_edge_compile_config,
)


class TestMul(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

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

    class MulScalar(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.mul.Scalar(x, 0.5)

    class MulRelu(torch.nn.Module):
        def forward(self, x, y):
            z = x * y
            return torch.nn.functional.relu(z)

    def _test_mul(self, inputs):
        (
            Tester(self.Mul(), inputs)
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_mul_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_mul(self):
        inputs = (
            torch.randn((1, 3)).to(torch.float16),
            torch.randn((4, 3)).to(torch.float16),
        )
        self._test_mul(inputs)

    def test_fp32_mul(self):
        inputs = (torch.randn((1, 3)), torch.randn((4, 3)))
        self._test_mul(inputs)

    def test_fp32_mul_scalar(self):
        (
            Tester(self.MulScalar(), (torch.randn(2, 3),))
            .export()
            .to_edge_transform_and_lower(
                transform_passes=get_transform_passes(),
                edge_compile_config=get_xnnpack_edge_compile_config(
                    skip_dim_order=True
                ),
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "executorch_exir_dialects_edge__ops_aten_mul_Scalar",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_mul(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1))
        (
            Tester(self.Mul(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_mul2(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        (
            Tester(self.Mul2(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_mul_functional(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.MulFunctional(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.mul.Tensor": 3})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_mul_relu(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.MulRelu(), inputs)
            .quantize()
            .export()
            .check_count(
                {
                    "torch.ops.aten.mul.Tensor": 1,
                    "torch.ops.aten.relu.default": 1,
                }
            )
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
                    "executorch_exir_dialects_edge__ops_aten_relu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
