# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestMinimum(unittest.TestCase):
    class Minimum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.minimum(x, y)

    def test_fp32_minimum(self):
        inputs = (
            torch.randn(1, 3, 6),
            torch.randn(1, 3, 6),
        )
        (
            Tester(self.Minimum(), inputs)
            .export()
            .check_count({"torch.ops.aten.minimum.default": 1})
            .to_edge()
            .check_count({"executorch_exir_dialects_edge__ops_aten_minimum_default": 1})
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_minimum_default"])
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
