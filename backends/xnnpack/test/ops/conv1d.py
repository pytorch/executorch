# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.xnnpack.test.tester import Tester


class TestConv1d(unittest.TestCase):
    class Conv1d(torch.nn.Module):
        def __init__(self):
            groups = 1
            stride = [2]
            padding = [1]
            dilation = [1]
            in_channels = 2
            out_channels = 1
            kernel_size = (3,)

            super().__init__()

            self.conv1d = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=True,
            )

        def forward(self, x):
            return self.conv1d(x)

    def test_conv1d(self):
        inputs = (torch.randn(1, 2, 4),)
        (
            Tester(self.Conv1d(), inputs)
            .export()
            .check_count({"torch.ops.aten.convolution.default": 1})
            .to_edge()
            .check_count(
                {"executorch_exir_dialects_edge__ops_aten_convolution_default": 1}
            )
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method()
            .compare_outputs()
        )
