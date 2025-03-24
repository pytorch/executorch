# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestDiv(unittest.TestCase):
    class Div(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = x / y
            return z

    class DivSingleInput(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = x / x
            return z

    def _test_div(self, inputs, mixed_dtype=False):
        tester = (
            Tester(self.Div(), inputs)
            .export()
            .check_count({"torch.ops.aten.div.Tensor": 1})
            .to_edge_transform_and_lower()
        )

        if mixed_dtype:
            # Inverse check for mixed-dtype: original node remains and no delegate node
            tester.check_count({"executorch_exir_dialects_edge__ops_aten_div_Tensor": 1})
            tester.check_not(["torch.ops.higher_order.executorch_call_delegate"])
        else:
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not(["executorch_exir_dialects_edge__ops_aten_div_Tensor"])

        (
            tester.to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_div(self):
        # Adding 4 to move distribution away from 0, 4 Std Dev should be far enough
        inputs = (
            (torch.randn(1) + 4).to(torch.float16),
            (torch.randn(1) + 4).to(torch.float16),
        )
        self._test_div(inputs)

    def test_fp32_div(self):
        # Adding 4 to move distribution away from 0, 4 Std Dev should be far enough
        inputs = (torch.randn(1) + 4, torch.randn(1) + 4)
        self._test_div(inputs)

    def test_fp32_div_single_input(self):
        # Adding 4 to move distribution away from 0, 4 Std Dev should be far enough
        inputs = (torch.randn(1) + 4,)
        (
            Tester(self.DivSingleInput(), inputs)
            .export()
            .check_count({"torch.ops.aten.div.Tensor": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_div_Tensor"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_div_with_mixed_dtype(self):
        test_cases = [
            torch.bfloat16,
            torch.float16,
            torch.int8,
        ]
        for dtype in test_cases:
            with self.subTest(dtype=str(dtype)):
                inputs = (
                    (torch.randn(1) + 4).to(torch.float32),
                    (torch.randn(1) + 4).to(dtype),
                )
                # Set mixed_dtype=True to verify that
                # no delegate node is inserted and the original node remains in the graph
                self._test_div(inputs, mixed_dtype=True)