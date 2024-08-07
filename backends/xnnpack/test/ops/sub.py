# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestSub(unittest.TestCase):
    class Sub(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x - y
            return z

    class Sub2(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = x - x
            return z

    def _test_sub(self, inputs):
        (
            Tester(self.Sub(), inputs)
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_sub_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_sub(self):
        inputs = (
            torch.randn((1, 3)).to(torch.float16),
            torch.randn((4, 3)).to(torch.float16),
        )
        self._test_sub(inputs)

    def test_fp32_sub(self):
        inputs = (torch.randn((1, 3)), torch.randn((4, 3)))
        self._test_sub(inputs)

    @unittest.skip("T171957656 - Quantized sub not implemented.")
    def _test_qs8_sub(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.Sub(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("T171957656 - Quantized sub not implemented.")
    def _test_qs8_sub2(self):
        inputs = (torch.randn(1, 1, 4, 4),)
        (
            Tester(self.Sub2(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("T171957656 - Quantized sub not implemented.")
    def _test_qs8_sub3(self):
        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1))
        (
            Tester(self.Sub(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.sub.Tensor": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("T171957656 - Quantized sub not implemented.")
    def _test_qs8_sub_relu(self):
        class Sub(torch.nn.Module):
            def forward(self, x, y):
                z = x - y
                return torch.nn.functional.relu(z)

        inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        (
            Tester(self.Sub(), inputs)
            .quantize()
            .export()
            .check_count(
                {
                    "torch.ops.aten.sub.Tensor": 1,
                    "torch.ops.aten.relu.default": 1,
                }
            )
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_sub_Tensor",
                    "executorch_exir_dialects_edge__ops_aten_relu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
