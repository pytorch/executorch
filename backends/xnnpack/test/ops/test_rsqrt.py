# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestRsqrt(unittest.TestCase):
    class Rsqrt(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = torch.abs(x)
            z = torch.rsqrt(x)
            return z

    def _test_rsqrt(self, inputs):
        (
            Tester(self.Rsqrt(), inputs)
            .export()
            .check_count({"torch.ops.aten.rsqrt.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_rsqrt_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_rsqrt(self):
        inputs = (torch.randn(20).to(torch.float16),)
        self._test_rsqrt(inputs)

    def test_fp32_rsqrt(self):
        inputs = (torch.randn(20),)
        self._test_rsqrt(inputs)
