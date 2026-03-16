# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestSin(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Sin(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = torch.sin(x)
            return z

    def _test_sin(self, inputs, legacy_mode: bool = False):
        tester = (
            Tester(self.Sin(), inputs)
            .export()
            .check_count({"torch.ops.aten.sin.default": 1})
        )

        if legacy_mode:
            tester = tester.to_edge().partition()
        else:
            tester = tester.to_edge_transform_and_lower()

        (
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_sin_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_sin(self):
        inputs = (
            torch.Tensor(
                [
                    [0.0, 0.1, 0.5, 0.785398],
                    [-0.5, -0.785398, 1.5708, -1.5708],
                ],
            ).to(torch.float16),
        )
        self._test_sin(inputs, legacy_mode=False)

    def test_fp16_sin_legacy_mode(self):
        inputs = (
            torch.Tensor(
                [
                    [0.0, 0.1, 0.5, 0.785398],
                    [-0.5, -0.785398, 1.5708, -1.5708],
                ],
            ).to(torch.float16),
        )
        self._test_sin(inputs, legacy_mode=True)

    def test_fp32_sin(self):
        inputs = (
            torch.Tensor(
                [
                    [0.0, 0.1, 0.5, 0.785398],
                    [-0.5, -0.785398, 1.5708, -1.5708],
                ],
            ),
        )
        self._test_sin(inputs, legacy_mode=False)

    def test_fp32_sin_legacy_mode(self):
        inputs = (
            torch.Tensor(
                [
                    [0.0, 0.1, 0.5, 0.785398],
                    [-0.5, -0.785398, 1.5708, -1.5708],
                ],
            ),
        )
        self._test_sin(inputs, legacy_mode=True)
