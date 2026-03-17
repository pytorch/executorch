# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from typing import Optional

import torch

try:
    import executorch.extension.pybindings.portable_lib  # noqa[F401]
    import executorch.kernels.quantized  # noqa[F401]

    has_quantized_ops = True
except:
    has_quantized_ops = False

from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)
from executorch.backends.xnnpack.test.test_xnnpack_utils import randomize_bn
from executorch.backends.xnnpack.test.tester import Quantize, Tester
from executorch.backends.xnnpack.test.tester.tester import ToEdgeTransformAndLower
from executorch.exir.dialects._ops import ops as exir_ops


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
        transpose=False,
    ):
        super().__init__()
        self.batches = batches
        self.width = width
        self.height = height
        self.in_channels = in_channels
        self.dtype = dtype

        op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self.transpose = transpose
        self.conv = op(
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
    def __init__(self, transpose=False):
        super().__init__()
        op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self.transpose = transpose
        self.first = op(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            padding=1,
            bias=False,
        )
        self.second = op(
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
    def __init__(self, transpose=False):
        super().__init__()
        op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self.transpose = transpose
        self.conv1 = op(
            2,
            2,
            (2, 2),
            bias=False,
            padding=[1, 1],
            stride=[4, 4],
        )
        self.bn = randomize_bn(2)
        self.hardtanh = torch.nn.Hardtanh()
        self.conv2 = op(
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
    def __init__(self, permute_order, transpose=False):
        super().__init__()
        op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self.transpose = transpose
        self.conv = op(
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


class Conv2dDQSeq(torch.nn.Module):
    def __init__(self, transpose=False):
        super().__init__()
        op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self.first = op(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.second = op(in_channels=8, out_channels=10, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.first(x)
        return self.second(y)

    def get_inputs(self):
        return (torch.randn(1, 3, 8, 8),)


class Conv2dDQParallel(torch.nn.Module):
    def __init__(self, transpose=False):
        super().__init__()
        op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self.first = op(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.second = op(in_channels=3, out_channels=10, kernel_size=3, padding=1)

    def forward(self, x):
        first = self.first(x)
        second = self.second(x)
        return first, second

    def get_inputs(self):
        return (torch.randn(1, 3, 8, 8),)


class TestConv2d(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def _test(
        self,
        m: torch.nn.Module,
        quant_config: Optional[QuantizationConfig] = None,
        conv_count=1,
        dtype: torch.dtype = torch.float,
        check_quantized=True,
    ):
        # pyre-fixme[29]: `Union[torch._tensor.Tensor,
        #  torch.nn.modules.module.Module]` is not a function.
        tester = Tester(m.eval(), m.get_inputs())

        if quant_config is not None:
            tester = tester.quantize(Quantize(quantization_config=quant_config))
            if check_quantized:
                tester.check(["torch.ops.quantized_decomposed"])

        op = (
            "torch.ops.aten.conv2d"
            if not m.transpose
            else "torch.ops.aten.conv_transpose2d"
        )

        (tester.export().check_count({op: conv_count}).to_edge_transform_and_lower())

        (
            tester.check_not(
                ["executorch_exir_dialects_edge__ops_aten_convolution_default"]
            )
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

    def _test_dq(
        self,
        m: torch.nn.Module,
        conv_count=1,
        dynamic_shapes=None,
    ):
        quant_config = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
        )

        DynamicallyQuantizedPartitioner = XnnpackPartitioner(
            config_precisions=ConfigPrecisionType.DYNAMIC_QUANT, per_op_mode=True
        )

        tester = Tester(m, m.get_inputs(), dynamic_shapes=dynamic_shapes)
        tester.quantize(Quantize(quantization_config=quant_config))
        tester.export()
        tester.check(["torch.ops.quantized_decomposed.choose_qparams"])
        tester.to_edge_transform_and_lower(
            ToEdgeTransformAndLower([DynamicallyQuantizedPartitioner])
        )
        tester.check_count(
            {"torch.ops.higher_order.executorch_call_delegate": conv_count}
        )
        tester.check_not(["executorch_exir_dialects_edge__ops_aten_conv2d_default"])
        tester.to_executorch()
        tester.serialize()
        tester.run_method_and_compare_outputs(qtol=1)

    def test_fp16_conv2d(self) -> None:
        for transpose in (True, False):
            for has_bias in (True, False):
                self._test(
                    Conv2d(bias=has_bias, dtype=torch.float16, transpose=transpose)
                )

    def test_fp32_conv2d(self) -> None:
        for transpose in (True, False):
            for has_bias in (True, False):
                self._test(Conv2d(bias=has_bias, transpose=transpose))

    def test_fp32_conv2d_permute(self) -> None:
        for transpose in (True, False):
            for perm_order in list(itertools.permutations([0, 1, 2, 3])):
                self._test(Conv2dPermute(perm_order, transpose=transpose))

    def test_qs8_conv2d_test(self) -> None:
        for transpose in (True, False):
            for has_bias in (True, False):
                self._test(
                    Conv2d(bias=has_bias, transpose=transpose),
                    quant_config=get_symmetric_quantization_config(),
                )

    def test_qs8_conv2d_per_channel(self) -> None:
        for transpose in (True, False):
            self._test(
                Conv2d(transpose=transpose),
                quant_config=get_symmetric_quantization_config(is_per_channel=True),
            )

    def test_fp32_conv2d_seq(self) -> None:
        for transpose in (True, False):
            self._test(Conv2dSeq(transpose=transpose), conv_count=2)

    def test_qs8_conv2d_seq(self) -> None:
        for transpose in (True, False):
            self._test(
                Conv2dSeq(transpose=transpose),
                conv_count=2,
                quant_config=get_symmetric_quantization_config(),
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
        for transpose in (True, False):
            self._test(
                Conv2d(groups=2, in_channels=2, out_channels=6, transpose=transpose)
            )

    def test_qs8_conv2d_depthwise(self):
        self._test(
            Conv2d(groups=2, in_channels=2, out_channels=6),
            quant_config=get_symmetric_quantization_config(),
        )

    def test_fp32_conv2d_bn(self):
        class Conv2dBatchNorm(torch.nn.Module):
            def __init__(
                self, in_features: int, out_features: int, kernel_size, transpose=False
            ):
                super().__init__()
                op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
                self.transpose = transpose
                self.conv2d = op(in_features, out_features, kernel_size)
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

        for transpose in (True, False):
            self._test(
                Conv2dBatchNorm(
                    in_features=2,
                    out_features=2,
                    kernel_size=(2, 2),
                    transpose=transpose,
                )
            )

    def test_fp32_conv2d_bn_hardtanh_mean_sequence(self):
        """
        This test makes sure that we can fuse batchnorm and hardtanh
        even with inserting copy nodes at some spots in the graph to change
        memory format
        """

        class Conv2dBatchNormHardTanh(torch.nn.Module):
            def __init__(
                self, in_channels: int, out_channels: int, kernel_size, transpose=False
            ):
                super().__init__()
                op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
                self.transpose = transpose
                self.conv = op(
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

        for transpose in (True, False):
            self._test(
                Conv2dBatchNormHardTanh(
                    in_channels=2,
                    out_channels=1,
                    kernel_size=(2, 2),
                    transpose=transpose,
                )
            )

    def test_qs8_conv2d_bn(self):
        for transpose in (True, False):
            self._test(
                Conv2dBatchNorm(transpose=transpose),
                quant_config=get_symmetric_quantization_config(),
                conv_count=2,
            )

    def test_qs8_conv2d_relu(self):
        class ConvReLU(torch.nn.Module):
            def __init__(self, transpose=False):
                super().__init__()
                op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
                self.transpose = transpose
                self.conv1 = op(
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

        for transpose in (True, False):
            self._test(
                ConvReLU(transpose=transpose),
                quant_config=get_symmetric_quantization_config(is_per_channel=True),
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
            def __init__(self, transpose=False):
                super().__init__()
                op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
                self.transpose = transpose
                self.conv1 = op(
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
            def __init__(self, transpose=False):
                super().__init__()
                op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
                self.transpose = transpose
                self.model = torch.nn.Sequential(
                    op(1, 1, 1),
                    torch.nn.ReLU(),
                    op(1, 64, 1),
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                return self.model(x)

            def get_inputs(self):
                return (torch.randn(1, 1, 1, 1),)

        for transpose in (True, False):
            self._test(
                ConvReLUSeq(transpose=transpose),
                quant_config=get_symmetric_quantization_config(),
                conv_count=2,
            )

    def test_qs8_conv2d_relu_multi_users(self):
        class Conv2dReluMultiUsers(torch.nn.Module):
            def __init__(self, transpose=False):
                super().__init__()
                op = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
                self.transpose = transpose
                self.conv1 = op(1, 1, 1)
                self.conv2 = op(1, 64, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                conv_default = self.conv1(x)
                y = self.relu(conv_default)
                conv_default_2 = self.conv2(y)
                return conv_default + conv_default_2

            def get_inputs(self):
                return (torch.randn(1, 1, 1, 1),)

        for transpose in (True, False):
            self._test(
                Conv2dReluMultiUsers(transpose=transpose),
                quant_config=get_symmetric_quantization_config(),
                conv_count=2,
            )

    def test_qs8_conv_transpose_2d_quantize_per_channel_multi_axis(self):
        class PerChannelConvTranspose2d(torch.nn.Module):
            def __init__(self, input_channels, output_channels, groups, axis):
                super().__init__()
                self.input_channels = input_channels
                self.output_channels = output_channels
                self.axis = axis
                self.groups = groups
                self.transpose = True
                self.weights = torch.nn.Parameter(
                    torch.randint(
                        low=-127,
                        high=127,
                        size=(input_channels, output_channels // groups, 4, 4),
                    ).type(dtype=torch.int8),
                    requires_grad=False,
                )

                axis_size = self.weights.shape[axis]
                self.scale = torch.nn.Parameter(torch.ones(axis_size) * 0.12345)
                self.zero_point = torch.nn.Parameter(
                    torch.zeros((axis_size,), dtype=torch.int64), requires_grad=False
                )

            def forward(self, x):
                dequantize_weights = (
                    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default(
                        self.weights,
                        self.scale,
                        self.zero_point,
                        self.axis,
                        -127,
                        127,
                        torch.int8,
                    )
                )
                dequantize_input = (
                    exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                        x, 0.12345, 0, -127, 127, torch.int8
                    )
                )
                x = torch.nn.functional.conv_transpose2d(
                    dequantize_input, dequantize_weights, groups=self.groups
                )

                return exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                    exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                        x,
                        0.12345,
                        0,
                        -127,
                        127,
                        torch.int8,
                    ),
                    0.12345,
                    0,
                    -127,
                    127,
                    torch.int8,
                )

            def get_inputs(self):
                return (
                    torch.randint(
                        low=-127, high=127, size=(3, self.input_channels, 4, 4)
                    ).type(dtype=torch.int8),
                )

        for groups in (1, 2):
            for ch_axis in (1, 2):
                if ch_axis == 1 and groups == 1:
                    self._test(
                        PerChannelConvTranspose2d(
                            3 * groups, 5 * groups, groups, ch_axis
                        ),  # ch_axis=0
                        quant_config=None,
                        conv_count=1,
                    )
                else:
                    with self.assertRaises(RuntimeError):
                        self._test(
                            PerChannelConvTranspose2d(
                                3 * groups, 5 * groups, groups, ch_axis
                            ),  # ch_axis=0
                            quant_config=None,
                            conv_count=1,
                        )

    def test_fp32_tconv_output_padding(self):
        """Test transposed convolution with non-zero output padding."""

        class TConv2dOutputPadding(torch.nn.Module):
            def __init__(self, output_padding):
                super().__init__()
                self.transpose = True
                self.conv = torch.nn.ConvTranspose2d(
                    in_channels=2,
                    out_channels=1,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    output_padding=output_padding,
                    dilation=(1, 1),
                    groups=1,
                    bias=True,
                ).to(torch.float)

            def forward(self, x):
                return self.conv(x)

            def get_inputs(self):
                return (torch.randn(1, 2, 8, 8),)

        # Test asymmetric output padding (0, 1)
        self._test(TConv2dOutputPadding(output_padding=(0, 1)))

        # Test symmetric output padding (1, 1)
        self._test(TConv2dOutputPadding(output_padding=(1, 1)))

    def test_qs8_tconv_output_padding(self):
        """Test quantized transposed convolution with non-zero output padding."""

        class TConv2dOutputPadding(torch.nn.Module):
            def __init__(self, output_padding):
                super().__init__()
                self.transpose = True
                self.conv = torch.nn.ConvTranspose2d(
                    in_channels=2,
                    out_channels=1,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    output_padding=output_padding,
                    dilation=(1, 1),
                    groups=1,
                    bias=True,
                ).to(torch.float)

            def forward(self, x):
                return self.conv(x)

            def get_inputs(self):
                return (torch.randn(1, 2, 8, 8),)

        # Test asymmetric output padding (0, 1) with quantization
        self._test(
            TConv2dOutputPadding(output_padding=(0, 1)),
            quant_config=get_symmetric_quantization_config(),
        )

        # Test symmetric output padding (1, 1) with quantization
        self._test(
            TConv2dOutputPadding(output_padding=(1, 1)),
            quant_config=get_symmetric_quantization_config(),
        )

    def test_fp32_tconv_output_padding_large_stride(self):
        """Test transposed convolution with larger output padding and stride values."""

        class TConv2dLargeOutputPadding(torch.nn.Module):
            def __init__(self, stride, output_padding):
                super().__init__()
                self.transpose = True
                self.conv = torch.nn.ConvTranspose2d(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=(5, 5),
                    stride=stride,
                    padding=(2, 2),
                    output_padding=output_padding,
                    dilation=(1, 1),
                    groups=1,
                    bias=True,
                ).to(torch.float)

            def forward(self, x):
                return self.conv(x)

            def get_inputs(self):
                return (torch.randn(2, 8, 16, 16),)

        # Test with stride=4 and output_padding=(3, 3) - maximum valid for stride 4
        self._test(TConv2dLargeOutputPadding(stride=(4, 4), output_padding=(3, 3)))

        # Test with stride=3 and asymmetric output_padding=(2, 1)
        self._test(TConv2dLargeOutputPadding(stride=(3, 3), output_padding=(2, 1)))

        # Test with asymmetric stride and output_padding
        self._test(TConv2dLargeOutputPadding(stride=(4, 3), output_padding=(3, 2)))

    def test_fp32_tconv_output_padding_various_shapes(self):
        """Test transposed convolution with output padding on various input shapes."""

        class TConv2dVariousShapes(torch.nn.Module):
            def __init__(
                self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                height,
                width,
            ):
                super().__init__()
                self.transpose = True
                self.height = height
                self.width = width
                self.in_channels = in_channels
                self.conv = torch.nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=(1, 1),
                    groups=1,
                    bias=True,
                ).to(torch.float)

            def forward(self, x):
                return self.conv(x)

            def get_inputs(self):
                return (torch.randn(1, self.in_channels, self.height, self.width),)

        # Test with larger kernel (7x7), stride=2, output_padding=1
        self._test(
            TConv2dVariousShapes(
                in_channels=3,
                out_channels=32,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                output_padding=(1, 1),
                height=32,
                width=32,
            )
        )

        # Test with rectangular kernel and asymmetric output padding
        self._test(
            TConv2dVariousShapes(
                in_channels=16,
                out_channels=8,
                kernel_size=(3, 5),
                stride=(2, 3),
                padding=(1, 2),
                output_padding=(1, 2),
                height=24,
                width=32,
            )
        )

        # Test with small spatial dimensions but larger output padding
        self._test(
            TConv2dVariousShapes(
                in_channels=4,
                out_channels=4,
                kernel_size=(4, 4),
                stride=(4, 4),
                padding=(0, 0),
                output_padding=(3, 3),
                height=4,
                width=4,
            )
        )

        # Test with batch size > 1 and asymmetric everything
        model = TConv2dVariousShapes(
            in_channels=6,
            out_channels=12,
            kernel_size=(5, 3),
            stride=(3, 2),
            padding=(2, 1),
            output_padding=(2, 1),
            height=20,
            width=15,
        )
        # Override get_inputs to use larger batch
        model.get_inputs = lambda: (torch.randn(4, 6, 20, 15),)
        self._test(model)

    def test_dq_conv2d(self) -> None:
        model = Conv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            batches=1,
            width=8,
            height=8,
        )
        self._test_dq(model)

    def test_dq_conv2d_seq(self) -> None:
        model = Conv2dDQSeq()
        conv_count = sum(1 for m in model.modules() if type(m) is torch.nn.Conv2d)
        self._test_dq(model, conv_count)

    def test_dq_conv2d_parallel(self) -> None:
        model = Conv2dDQParallel()
        conv_count = sum(1 for m in model.modules() if type(m) is torch.nn.Conv2d)
        self._test_dq(model, conv_count)

    def test_dq_conv2d_transpose(self) -> None:
        model = Conv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            batches=1,
            width=8,
            height=8,
            transpose=True,
        )
        self._test_dq(model)

    def test_dq_conv2d_transpose_seq(self) -> None:
        model = Conv2dDQSeq(transpose=True)
        conv_count = sum(
            1 for m in model.modules() if type(m) is torch.nn.ConvTranspose2d
        )
        self._test_dq(model, conv_count)

    def test_dq_conv2d_transpose_parallel(self) -> None:
        model = Conv2dDQParallel(transpose=True)
        conv_count = sum(
            1 for m in model.modules() if type(m) is torch.nn.ConvTranspose2d
        )
        self._test_dq(model, conv_count)
