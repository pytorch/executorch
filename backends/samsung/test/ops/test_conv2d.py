# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.


import unittest

import torch

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=16,
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        bias=True,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ).to(torch.float)

        self.in_channels = in_channels

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        input_1 = torch.randn(1, self.in_channels, 24, 24)
        return (input_1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TransposeConv2d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels=32,
            out_channels=8,
            kernel_size=2,
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            bias=True,
        )

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        input_1 = torch.randn(1, 32, 24, 24)
        return (input_1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestConv2d(unittest.TestCase):
    def _test(self, module: torch.nn.Module):
        tester = SamsungTester(
            module,
            module.get_example_inputs(),
            [gen_samsung_backend_compile_spec("E9955")],
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    def test_fp32_conv2d_without_bias(self):
        self._test(Conv2d(bias=False))

    def test_fp32_conv2d_with_bias(self):
        self._test(Conv2d(bias=True))

    def test_fp32_depthwise_conv2d(self):
        self._test(Conv2d(in_channels=8, out_channels=8, groups=8))

    def test_fp32_transpose_conv2d(self):
        self._test(TransposeConv2d())
