# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestAdd(unittest.TestCase):
    class Add(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x + y
            z = z + x
            z = z + x
            z = z + z
            return z

    class Add2(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = x + x
            return z

    class AddConstant(torch.nn.Module):
        def __init__(self, constant):
            super().__init__()
            self._constant1 = constant
            self.register_buffer("_constant2", constant, persistent=False)
            self.register_parameter("_constant3", torch.nn.Parameter(constant))

        def forward(self, x):
            out1 = x + self._constant1 + torch.ones(1, 1, 1)
            out2 = x + self._constant2 + self._constant3
            return out1, out2

    def _test_add(self, inputs):
        (
            Tester(self.Add(), inputs)
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_add(self):
        inputs = (torch.randn(1).to(torch.float16), torch.randn(1).to(torch.float16))
        self._test_add(inputs)

    def test_fp32_add(self):
        inputs = (torch.randn(1), torch.randn(1))
        self._test_add(inputs)

    def test_fp32_add_constant(self):
        inputs = (torch.randn(4, 4, 4),)
        (
            Tester(self.AddConstant(torch.randn(4, 4, 4)), inputs)
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_add_constant(self):
        inputs = (torch.randn(4, 4, 4),)
        (
            Tester(self.AddConstant(torch.randn(4, 4, 4)), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_add(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.Add(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_add2(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        (
            Tester(self.Add2(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_add3(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1))
        (
            Tester(self.Add(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 4})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_add_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class AddRelu(torch.nn.Module):
        def forward(self, x, y):
            z = x + y
            return torch.nn.functional.relu(z)

    def test_fp32_add_relu(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.AddRelu(), inputs)
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check_count({"torch.ops.aten.relu.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_add_Tensor"])
            .check_not(["executorch_exir_dialects_edge__ops_aten_relu_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_add_relu(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.AddRelu(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.add.Tensor": 1})
            .check_count({"torch.ops.aten.relu.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_add_relu_seq(self):
        class AddReLU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x, z):
                y = x + z
                y = self.relu(y)
                y = y + y
                y = self.relu(y)
                return y

        inputs = (
            torch.randn(
                1,
                1,
                20,
                20,
            ),
            torch.randn(
                1,
                1,
                20,
                20,
            ),
        )

        (
            Tester(self.AddRelu(), inputs)
            .quantize()
            .export()
            .check_count(
                {"torch.ops.aten.add.Tensor": 1, "torch.ops.aten.relu.default": 1}
            )
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
