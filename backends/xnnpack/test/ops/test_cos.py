# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestCos(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class Cos(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            z = torch.cos(x)
            return z

    def _test_cos(self, inputs, legacy_mode: bool = False, atol: float = 1e-4):
        tester = (
            Tester(self.Cos(), inputs)
            .export()
            .check_count({"torch.ops.aten.cos.default": 1})
        )

        if legacy_mode:
            tester = tester.to_edge().partition()
        else:
            tester = tester.to_edge_transform_and_lower()

        (
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_cos_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=atol)
        )

    def test_fp16_cos(self):
        inputs = (
            torch.Tensor(
                [
                    [0.0, 0.1, 0.5, 0.785398],
                    [-0.5, -0.785398, 1.5708, -1.5708],
                ],
            ).to(torch.float16),
        )
        self._test_cos(inputs, legacy_mode=False, atol=2e-3)

    def test_fp16_cos_legacy_mode(self):
        inputs = (
            torch.Tensor(
                [
                    [0.0, 0.1, 0.5, 0.785398],
                    [-0.5, -0.785398, 1.5708, -1.5708],
                ],
            ).to(torch.float16),
        )
        self._test_cos(inputs, legacy_mode=True, atol=2e-3)

    def test_fp32_cos(self):
        inputs = (
            torch.Tensor(
                [
                    [0.0, 0.1, 0.5, 0.785398],
                    [-0.5, -0.785398, 1.5708, -1.5708],
                ],
            ),
        )
        self._test_cos(inputs, legacy_mode=False)

    def test_fp32_cos_legacy_mode(self):
        inputs = (
            torch.Tensor(
                [
                    [0.0, 0.1, 0.5, 0.785398],
                    [-0.5, -0.785398, 1.5708, -1.5708],
                ],
            ),
        )
        self._test_cos(inputs, legacy_mode=True)
