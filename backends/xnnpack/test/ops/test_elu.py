# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestElu(unittest.TestCase):
    class ELU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.elu = torch.nn.ELU(alpha=0.5)

        def forward(self, x):
            return self.elu(x)

    class ELUFunctional(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.elu(x, alpha=1.2)

    def _test_elu(self, inputs):
        (
            Tester(self.ELU(), inputs)
            .export()
            .check_count({"torch.ops.aten.elu.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_elu_default",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("PyTorch Pin Update Required")
    def _test_fp16_elu(self):
        inputs = (torch.randn(1, 3, 3).to(torch.float16),)
        self._test_elu(inputs)

    @unittest.skip("PyTorch Pin Update Required")
    def _test_fp32_elu(self):
        inputs = (torch.randn(1, 3, 3),)
        self._test_elu(inputs)

    @unittest.skip("Update Quantizer to quantize Elu")
    def _test_qs8_elu(self):
        inputs = (torch.randn(1, 3, 4, 4),)
        (
            Tester(self.ELU(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.elu.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_elu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    @unittest.skip("Update Quantizer to quantize Elu")
    def _test_qs8_elu_functional(self):
        inputs = (torch.randn(1, 3, 4, 4),)
        (
            Tester(self.ELU(), inputs)
            .quantize()
            .export()
            .check_count({"torch.ops.aten.elu.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_elu_default",
                    "torch.ops.quantized_decomposed",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
