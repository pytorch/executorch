# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestMaximum(unittest.TestCase):
    class Maximum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.maximum(x, y)

    def _test_maximum(self, inputs):
        (
            Tester(self.Maximum(), inputs)
            .export()
            .check_count({"torch.ops.aten.maximum.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_maximum_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_maximum(self):
        inputs = (
            torch.randn(2, 3, 4).to(torch.float16),
            torch.randn(2, 3, 4).to(torch.float16),
        )
        self._test_maximum(inputs)

    def test_fp32_maximum(self):
        inputs = (
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
        )
        self._test_maximum(inputs)

    def test_fp32_maximum_broadcast(self):
        inputs = (
            torch.randn(2, 3, 4),
            torch.randn(2, 1, 4),
        )
        (
            Tester(self.Maximum(), inputs)
            .export()
            .check_count({"torch.ops.aten.maximum.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_maximum_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
