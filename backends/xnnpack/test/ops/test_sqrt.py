# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestSqrt(unittest.TestCase):
    class Sqrt(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.abs(x)
            z = torch.sqrt(x)
            return z

    def _test_sqrt(self, inputs):
        (
            Tester(self.Sqrt(), inputs)
            .export()
            .check_count({"torch.ops.aten.sqrt.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_sqrt_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_sqrt(self):
        inputs = (torch.randn(20).to(torch.float16),)
        self._test_sqrt(inputs)

    def test_fp32_sqrt(self):
        inputs = (torch.randn(20),)
        self._test_sqrt(inputs)
