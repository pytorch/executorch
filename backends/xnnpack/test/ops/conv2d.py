# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from typing import Optional

import torch
from executorch.backends.xnnpack.test.test_xnnpack_utils import randomize_bn
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
        dtype=torch.float,
    ):
        super().__init__()
        self.batches = batches
        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.dtype = dtype

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
        ).to(dtype)

    def forward(self, x):
        return self.conv(x)

    def get_inputs(self):
        return (
            torch.randn(self.batches, self.in_channels, self.height, self.width).to(
                self.dtype
            ),
        )


class Conv2dSeq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            padding=1,
            bias=False,
        )
        self.second = torch.nn.Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=(3, 3),
            padding=1,
            bias=False,
        )

    def forward(self, x):
        y = self.first(x)
        return self.second(y)

    def get_inputs(self):
        return (torch.randn(1, 1, 3, 3),)


class Conv2dBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            2,
            2,
            (2, 2),
            bias=False,
            padding=[1, 1],
            stride=[4, 4],
        )
        self.bn = randomize_bn(2)
        self.hardtanh = torch.nn.Hardtanh()
        self.conv2 = torch.nn.Conv2d(
            2,
            2,
            (2, 2),
            bias=False,
            padding=[1, 1],
            stride=[4, 4],
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn(y)
        y = self.hardtanh(y)
        y = self.conv2(y)
        y = self.bn(y)
        y = self.hardtanh(y)
        return y

    def get_inputs(self):
        return (torch.randn(2, 2, 4, 4),)


class Conv2dPermute(torch.nn.Module):
    def __init__(self, permute_order):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            2,
            2,
            (2, 2),
            bias=False,
            padding=[2, 2],
            stride=[2, 2],
        )
        self.permute_order = permute_order

    def forward(self, x):
        result = self.conv(x)
        channels_last = torch.permute(result, self.permute_order)
        return channels_last

    def get_inputs(self):
        return (torch.randn(2, 2, 4, 4),)


class TestConv2d(unittest.TestCase):
    def _test(
        self,
        m: torch.nn.Module,
        quant_config: Optional[QuantizationConfig] = None,
        conv_count=1,
        dtype: torch.dtype = torch.float,
    ):
        tester = Tester(m.eval(), m.get_inputs())

        if quant_config is not None:
            tester = tester.quantize(Quantize(quantization_config=quant_config))
            tester.check(["torch.ops.quantized_decomposed"])

        (
            tester.export()
            .check_count({"torch.ops.aten.conv2d": conv_count})
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops__native_batch_norm_legit_no_training_default"
                ]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(qtol=1)
        )

    def test_fp16_conv2d(self) -> None:
        for has_bias in (True, False):
            self._test(Conv2d(bias=has_bias, dtype=torch.float16))

    def test_fp32_conv2d(self) -> None:
        for has_bias in (True, False):
            self._test(Conv2d(bias=has_bias))

    def test_fp32_conv2d_permute(self) -> None:
        for perm_order in list(itertools.permutations([0, 1, 2, 3])):
            self._test(Conv2dPermute(perm_order))

    def test_qs8_conv2d_test(self) -> None:
        for has_bias in (True, False):
            self._test(
                Conv2d(bias=has_bias), quant_config=get_symmetric_quantization_config()
            )

    def test_qs8_conv2d_per_channel(self) -> None:
        self._test(
            Conv2d(),
            quant_config=get_symmetric_quantization_config(is_per_channel=True),
        )

    def test_fp32_conv2d_seq(self) -> None:
        self._test(Conv2dSeq(), conv_count=2)

    def test_qs8_conv2d_seq(self) -> None:
        self._test(
            Conv2dSeq(), conv_count=2, quant_config=get_symmetric_quantization_config()
        )

    def test_fp32_conv2d_single_int_params(self):
        self._test(
            Conv2d(
                kernel_size=3,
                stride=2,
                padding="valid",
                dilation=1,
            )
        )

    def test_fp32_conv2d_depthwise(self):
        # Depthwise Convolution Requirements:
        # - Groups must equal In Channels
        # - Out Channels must be a positive multiple of In Channels
        self._test(Conv2d(groups=2, in_channels=2, out_channels=6))

    def test_qs8_conv2d_depthwise(self):
        self._test(
            Conv2d(groups=2, in_channels=2, out_channels=6),
            quant_config=get_symmetric_quantization_config(),
        )

    def test_fp32_conv2d_bn(self):
        class Conv2dBatchNorm(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int, kernel_size):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(in_features, out_features, kernel_size)
                self.bn = randomize_bn(out_features)
                self.in_features = in_features
                self.kernel_size = kernel_size

            def forward(self, x):
                y = self.conv2d(x)
                y = self.bn(y)
                return y

            def get_inputs(self):
                return (
                    torch.randn(
                        2,
                        self.in_features,
                        self.kernel_size[0] * 2,
                        self.kernel_size[1] * 2,
                    ),
                )

        self._test(Conv2dBatchNorm(in_features=2, out_features=2, kernel_size=(2, 2)))

    def test_fp32_conv2d_bn_hardtanh_mean_sequence(self):
        """
        This test makes sure that we can fuse batchnorm and hardtanh
        even with inserting copy nodes at some spots in the graph to change
        memory format
        """

        class Conv2dBatchNormHardTanh(torch.nn.Module):
            def __init__(self, in_channels: int, out_channels: int, kernel_size):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=[1, 1],
                    stride=[2, 2],
                )
                self.in_channels = in_channels
                self.native_batchnorm = torch.nn.BatchNorm2d(out_channels)
                self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)

            def forward(self, x):
                x = self.conv(x)
                x = self.native_batchnorm(x)
                x = self.hardtanh(x)
                x = torch.mean(x, (-1, -2), keepdim=True)
                return x

            def get_inputs(self):
                return (torch.randn(2, self.in_channels, 8, 8),)

        self._test(
            Conv2dBatchNormHardTanh(in_channels=2, out_channels=1, kernel_size=(2, 2))
        )

    def test_qs8_conv2d_bn(self):
        self._test(
            Conv2dBatchNorm(),
            quant_config=get_symmetric_quantization_config(),
            conv_count=2,
        )

    def test_qs8_conv2d_relu(self):
        class ConvReLU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    2,
                    2,
                    (2, 2),
                    bias=False,
                    padding=[1, 1],
                    stride=[4, 4],
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.conv1(x)
                y = self.relu(y)
                return y

            def get_inputs(self):
                return (torch.randn(2, 2, 4, 4),)

        self._test(
            ConvReLU(),
            quant_config=get_symmetric_quantization_config(),
        )

    def test_qs8_conv2d_dw_relu(self):
        # Depthwise Convolution Requirements:
        # - Groups must equal In Channels
        # - Out Channels must be a positive multiple of In Channels
        groups = 2
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = groups
        out_channels = 3 * in_channels
        width = 8
        height = 8
        batches = 1

        class ModelConvReLU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    dilation=dilation,
                    bias=True,
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.conv1(x)
                y = self.relu(y)
                return y

            def get_inputs(self):
                return (torch.randn(batches, in_channels, height, width) * 11,)

        for per_channel_quant in (False, True):
            model = ModelConvReLU()
            self._test(
                model,
                quant_config=get_symmetric_quantization_config(
                    is_per_channel=per_channel_quant
                ),
            )

    def test_qs8_conv2d_relu_seq(self):
        class ConvReLUSeq(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(1, 64, 1),
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                return self.model(x)

            def get_inputs(self):
                return (torch.randn(1, 1, 1, 1),)

        self._test(
            ConvReLUSeq(),
            quant_config=get_symmetric_quantization_config(),
            conv_count=2,
        )
