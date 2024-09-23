# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester

from executorch.exir.dialects._ops import ops as exir_ops


class TestQuantizePerTensor(unittest.TestCase):
    def test_qs8_quantize_per_tensor(self):
        class Quant(torch.nn.Module):
            def forward(self, x):
                return exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                    x, 0.12345, 0, -127, 127, torch.int8
                )

        inputs = (torch.randn(1, 1, 4, 4),)
        (
            Tester(Quant(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default"
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_dequantize_per_tenstor(self):
        class Dequant(torch.nn.Module):
            def forward(self, x):
                return exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                    x, 0.12345, 0, -127, 127, torch.int8
                )

        inputs = (
            (
                torch.randint(low=-127, high=127, size=(1, 1, 4, 4)).type(
                    dtype=torch.int8
                )
            ),
        )

        (
            Tester(Dequant(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
