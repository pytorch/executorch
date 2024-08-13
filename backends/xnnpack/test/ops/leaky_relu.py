# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestLeakyRelu(unittest.TestCase):
    class LeakyReLU(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.relu = torch.nn.LeakyReLU(**kwargs)

        def forward(self, x):
            y = x + x
            z = self.relu(y)
            return z

    class LeakyReLUFunctional(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.leaky_relu(x)

    def _test_leaky_relu(self, module, inputs):
        (
            Tester(module, inputs)
            .export()
            .check_count({"torch.ops.aten.leaky_relu.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_leaky_relu_default",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_leaky_relu(self):
        inputs = (torch.randn(1, 3, 3).to(torch.float16),)
        module = self.LeakyReLUFunctional()
        self._test_leaky_relu(module, inputs)

    def test_fp32_leaky_relu(self):
        inputs = (torch.randn(1, 3, 3),)
        module = self.LeakyReLU(negative_slope=0.2)
        self._test_leaky_relu(module, inputs)

    def test_fp32_leaky_relu_functional(self):
        inputs = (torch.randn(1, 3, 3),)
        (
            Tester(self.LeakyReLUFunctional(), inputs)
            .export()
            .check_count({"torch.ops.aten.leaky_relu.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_leaky_relu_default",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("T172863987 - Missing quantizer support.")
    def _test_qs8_leaky_relu(self):
        inputs = (torch.randn(1, 3, 3),)
        (
            Tester(self.LeakyReLU(negative_slope=0.2), inputs)
            .quantize()
            .export()
            .check_node_count(
                {
                    "leaky_relu::default": 1,
                    "quantized_decomposed::quantize_per_tensor": 3,
                }
            )
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_leaky_relu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("T172863987 - Missing quantizer support.")
    def _test_qs8_leaky_relu_default_slope(self):
        """
        The leaky_relu visitor has logic to handle the default slope, since it's apparently not
        passed through on export. This test ensures that this matches the eager mode behavior.
        """

        inputs = (torch.randn(1, 3, 3),)
        (
            Tester(self.LeakyReLU(), inputs)
            .quantize()
            .export()
            .check_node_count(
                {
                    "leaky_relu::default": 1,
                    "quantized_decomposed::quantize_per_tensor": 3,
                }
            )
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_leaky_relu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
