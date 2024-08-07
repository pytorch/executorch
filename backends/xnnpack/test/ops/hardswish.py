# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestHardswish(unittest.TestCase):
    class Hardswish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hardswish = torch.nn.Hardswish()

        def forward(self, x):
            return self.hardswish(x)

    class HardswishFunctional(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.hardswish(x)

    def _test_hardswish(self, inputs):
        (
            Tester(self.Hardswish(), inputs)
            .export()
            .check_count({"torch.ops.aten.hardswish.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_hardswish_default",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_hardswish(self):
        inputs = (torch.randn(1, 3, 3).to(torch.float16),)
        self._test_hardswish(inputs)

    def test_fp32_hardswish(self):
        inputs = (torch.randn(1, 3, 3),)
        self._test_hardswish(inputs)

    def test_fp32_hardswish_functional(self):
        inputs = (torch.randn(1, 3, 3),)
        (
            Tester(self.HardswishFunctional(), inputs)
            .export()
            .check_count({"torch.ops.aten.hardswish.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_hardswish_default",
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
