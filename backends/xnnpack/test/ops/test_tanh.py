# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestTanh(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Tanh(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tanh(x)

    def run_tanh_test(self, inputs):
        (
            Tester(self.Tanh(), inputs)
            .export()
            .check_count({"torch.ops.aten.tanh.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_tanh_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_tanh(self):
        inputs = (torch.randn(20).to(torch.float16),)
        self.run_tanh_test(inputs)

    def test_fp32_tanh(self):
        inputs = (torch.randn(20),)
        self.run_tanh_test(inputs)
