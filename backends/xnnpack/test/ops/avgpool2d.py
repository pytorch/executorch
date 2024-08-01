# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestAvgPool2d(unittest.TestCase):
    class AvgPool2d(torch.nn.Module):
        def __init__(
            self, count_include_pad=False, ceil_mode=False, divisor_override=None
        ):
            super().__init__()
            self.avgPool = torch.nn.AvgPool2d(
                kernel_size=(2, 2),
                padding=(1, 1),
                stride=(2, 2),
                count_include_pad=count_include_pad,
                ceil_mode=ceil_mode,
                divisor_override=divisor_override,
            )

        def forward(self, x):
            return self.avgPool(x)

    def _test_argpool2d(self, inputs):
        (
            Tester(self.AvgPool2d(), inputs)
            .export()
            .check_count({"torch.ops.aten.avg_pool2d.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_avgpool2d(self):
        inputs = (torch.randn(1, 1, 10, 10).to(torch.float16),)
        self._test_argpool2d(inputs)

    def test_fp32_avgpool2d(self):
        inputs = (torch.randn(1, 1, 10, 10),)
        self._test_argpool2d(inputs)

    def test_fp32_avgpool2d_ceil_mode_unsupported(self):
        """
        The XNNPACK backend does not support ceil mode.
        """
        inputs = (torch.randn(1, 1, 10, 10),)
        (
            Tester(self.AvgPool2d(ceil_mode=True), inputs)
            .export()
            .check_count({"torch.ops.aten.avg_pool2d.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
        )

    def test_fp32_avgpool2d_count_include_pad_unsupported(self):
        """
        The XNNPACK backend does not support count_include_pad=True.
        """
        inputs = (torch.randn(1, 1, 10, 10),)
        (
            Tester(self.AvgPool2d(count_include_pad=True), inputs)
            .export()
            .check_count({"torch.ops.aten.avg_pool2d.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
        )

    def test_fp32_avgpool2d_divisor_override(self):
        """
        The XNNPACK backend does not support divisor overrides not equal to the pooling region.
        """
        inputs = (torch.randn(1, 1, 10, 10),)
        (
            Tester(self.AvgPool2d(divisor_override=5), inputs)
            .export()
            .check_count({"torch.ops.aten.avg_pool2d.default": 1})
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
        )
