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

    def _test_div(self, inputs):
        (
            Tester(self.Div(), inputs)
            .export()
            .check_count({"torch.ops.aten.div.Tensor": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_div_Tensor"])
            .to_executorch()
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
