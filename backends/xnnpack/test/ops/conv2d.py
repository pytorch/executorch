# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

import torch

from executorch.backends.xnnpack.test.tester import Quantize, Tester
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        bias=True,
        padding_mode="zeros",
        batches=1,
        width=8,
        height=8,
    ):
        super().__init__()
        self.batches = batches
        self.width = width
        self.height = height
        self.in_channels = in_channels

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.conv(x)

    def get_inputs(self):
        return (torch.randn(self.batches, self.in_channels, self.height, self.width),)


class TestConv2d(unittest.TestCase):
    def _test(
        self, m: torch.nn.Module, quant_config: Optional[QuantizationConfig] = None
    ):
        tester = Tester(m.eval(), m.get_inputs())

        if quant_config is not None:
            tester = tester.quantize(Quantize(quantization_config=quant_config))
            tester.check(["torch.ops.quantized_decomposed"])

        (
            tester.export()
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

    def test_conv2d(self) -> None:
        self._test(Conv2d())

    def test_qconv2d(self) -> None:
        self._test(Conv2d(), quant_config=get_symmetric_quantization_config())

    def test_qconv2d_per_channel(self) -> None:
        self._test(
            Conv2d(),
            quant_config=get_symmetric_quantization_config(is_per_channel=True),
        )
