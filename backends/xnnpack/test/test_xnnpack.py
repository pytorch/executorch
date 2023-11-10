# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.test_xnnpack_utils import (
    randomize_bn,
    TestXNNPACK,
)

from executorch.backends.xnnpack.test.test_xnnpack_utils_classes import (
    OpSequencesAddConv2d,
)


class TestXNNPACKFloatingPoint(TestXNNPACK):
    def test_xnnpack_backend_mean_dim(self):
        class Mean(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.mean(x, (-1, -2), keepdim=True)

        example_inputs = (torch.randn(1, 5, 4, 4),)
        self.lower_and_test_with_partitioner(Mean(), example_inputs)

    @unittest.expectedFailure
    def test_xnnpack_backend_mean_dim_unsupported(self):
        class Mean(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.mean(x, (3), keepdim=True)

        example_inputs = (torch.randn(1, 5, 4, 4),)
        self.lower_and_test_with_partitioner(Mean(), example_inputs)

    def test_xnnpack_backend_static_transpose(self):
        class PermuteModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nchw_to_nhwc = [0, 2, 3, 1]

            def forward(self, x):
                return torch.permute(x, self.nchw_to_nhwc)

        example_inputs = (torch.randn(1, 1, 4, 4),)
        self.lower_module_and_test_output(PermuteModule(), example_inputs)

    def test_xnnpack_backend_sequential_conv2d(self):
        class TwoConv(torch.nn.Module):
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
                return self.second(self.first(x))

        example_inputs = (torch.randn(1, 1, 3, 3),)
        self.lower_and_test_with_partitioner(TwoConv(), example_inputs)

    def test_xnnpack_backend_conv2d_bn(self):
        class ModelConvBN(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int, kernel_size):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(in_features, out_features, kernel_size)
                self.bn = randomize_bn(out_features)

            def forward(self, x):
                y = self.conv2d(x)
                y = self.bn(y)
                return y

        model = ModelConvBN(2, 2, (2, 2)).eval()
        self.lower_and_test_with_partitioner(model, (torch.randn(2, 2, 4, 4),))

    def test_xnnpack_backend_conv1d(self):
        groups = 1
        stride = [2]
        padding = [1]
        dilation = [1]
        in_channels = 2
        out_channels = 1
        kernel_size = (3,)
        height = 8
        batches = 1
        example_inputs = (torch.randn(batches, in_channels, height),)

        conv1 = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        conv1.eval()

        # Adjust parameters such that convolution output is same shape as input
        out_channels = in_channels
        stride = [1]

        class Conv1dBatchNormSequential(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    dilation=dilation,
                    bias=True,
                )
                self.bn1 = randomize_bn(num_features=in_channels, dimensionality=1)
                self.conv2 = torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    dilation=dilation,
                    bias=True,
                )
                self.bn2 = randomize_bn(num_features=in_channels, dimensionality=1)

            def forward(self, x):
                y = self.conv1(x)
                y = self.bn1(y)
                z = self.conv2(y)
                z = self.bn2(z)
                z = torch.add(y, z)
                return z

        conv2 = Conv1dBatchNormSequential()
        conv2.eval()

        self.lower_and_test_with_partitioner(conv1, example_inputs)
        self.lower_and_test_with_partitioner(conv2, example_inputs)

    def test_xnnpack_backend_conv2d(self):
        groups = 1
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 2
        out_channels = 1
        width = 8
        height = 8
        batches = 1
        example_inputs = (torch.randn(batches, in_channels, height, width),)
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        conv.eval()
        self.lower_and_test_with_partitioner(conv, example_inputs)

    def test_xnnpack_backend_conv2d_single_int_params(self):
        groups = 1
        stride = 2
        padding = "valid"
        dilation = 1
        in_channels = 2
        out_channels = 1
        width = 8
        height = 8
        batches = 1
        example_inputs = (torch.randn(batches, in_channels, height, width),)
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        conv.eval()
        self.lower_and_test_with_partitioner(conv, example_inputs)

    def test_xnnpack_backend_conv2d_dw(self):
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
        example_inputs = (torch.randn(batches, in_channels, height, width),)
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=True,
        )
        conv.eval()
        self.lower_and_test_with_partitioner(conv, example_inputs)

    @torch.inference_mode()  # TODO Use  for capturing.
    def test_xnnpack_backend_mm(self):
        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]
        for i, _ in enumerate(in_sizes):
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            linear = torch.nn.Linear(input_size, output_size, bias=False).eval()
            example_input = (torch.randn(in_size, input_size),)

            self.lower_and_test_with_partitioner(linear, example_input)

    def test_xnnpack_backend_addmm(self):
        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]
        for i, _ in enumerate(in_sizes):
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            linear = torch.nn.Linear(input_size, output_size, bias=True).eval()
            example_input = (torch.randn(in_size, input_size),)

            self.lower_and_test_with_partitioner(linear, example_input)

    def test_xnnpack_constant_add(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._constant = torch.ones(4, 4, 4)

            def forward(self, x):
                out1 = x + self._constant
                out2 = x + self._constant + self._constant
                return out1, out2

        const_module = Module()
        model_inputs = (torch.randn(4, 4, 4),)

        self.lower_and_test_with_partitioner(const_module, model_inputs)

    def test_xnnpack_backend_add(self):
        # This test is the simplest test by manually lowering some submodules, we can use paritioner for auto detecting lowerable parts
        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                z = z + z
                return z

        add_module = AddModule()
        model_inputs = (torch.ones(1), torch.ones(1))

        self.lower_and_test_with_partitioner(add_module, model_inputs)

    def test_xnnpack_backend_div(self):
        class DivModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x / y
                return z

        div_module = DivModule()
        model_inputs = (torch.ones(1), torch.ones(1))

        self.lower_and_test_with_partitioner(div_module, model_inputs)

    def test_xnnpack_minimum(self):
        class MinimumModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.minimum_module = torch.minimum

            def forward(self, x, y):
                return self.minimum_module(x, y)

        module = MinimumModule()
        model_inputs = (
            torch.randn(1, 3, 6),
            torch.randn(1, 3, 6),
        )
        self.lower_and_test_with_partitioner(module, model_inputs)

    @torch.inference_mode()  # TODO Use  for capturing.
    def test_xnnpack_backend_linear(self):
        in_size = 2
        input_size = 3
        output_size = 4
        linear = torch.nn.Linear(input_size, output_size).eval()
        example_input = (torch.randn(in_size, input_size),)

        self.lower_and_test_with_partitioner(linear, example_input)

    def test_xnnpack_backend_softmax(self):
        class SoftMaxModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.softmax = torch.nn.Softmax(dim=dim)

            def forward(self, x):
                return self.softmax(x)

        # We want to test that for tensor.dim() == 3 i.e. our test tensor,
        # values for softmax_dim = [-1, 2] passes and rest fails. This is because xnnpack
        # only supports softmax_dim == -1 i.e. the last dimension.
        shape = (3, 5, 7)
        for dim in list(range(len(shape))) + [-1]:
            model_inputs = (torch.rand(shape),)
            softmax_module = SoftMaxModule(dim)

            if dim == len(shape) - 1 or dim == -1:
                self.lower_and_test_with_partitioner(softmax_module, model_inputs)
            else:
                with self.assertRaises(RuntimeError):
                    self.lower_and_test_with_partitioner(softmax_module, model_inputs)

    def test_xnnpack_backend_hardtanh(self):
        class HardTanhModule(torch.nn.Module):
            def __init__(self, min_val=-1.0, max_val=1.0):
                super().__init__()
                self.hardtanh = torch.nn.Hardtanh(min_val, max_val)

            def forward(self, x):
                return self.hardtanh(x)

        inputs = [torch.randn(2, 3, 4), torch.randn(7, 5, 2), torch.randn(2, 9)]
        for test_input in inputs:
            hardtanh_model = HardTanhModule()
            self.lower_and_test_with_partitioner(hardtanh_model, (test_input,))

        for test_input in inputs:
            hardtanh_model = HardTanhModule(-2, 2)
            self.lower_and_test_with_partitioner(hardtanh_model, (test_input,))

    def test_xnnpack_backend_Relu(self):
        class ReluModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        example_input = torch.randn(2, 3, 4)
        self.lower_and_test_with_partitioner(ReluModule(), (example_input,))

    def test_xnnpack_backend_sigmoid(self):
        class SigmoidModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(x)

        model_inputs = (torch.rand(7, 5, 3),)
        sigmoid_module = SigmoidModule()
        self.lower_and_test_with_partitioner(sigmoid_module, model_inputs)

    # FIXME (T148779166)
    @unittest.expectedFailure
    def test_xnnpack_backend_static_resize_bilinear_2d(self):
        class StaticResizeBilinear2DModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                a = torch.nn.functional.interpolate(
                    x,
                    size=(x.shape[2] * 2, x.shape[3] * 3),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                a = torch.nn.functional.interpolate(
                    a,
                    scale_factor=3.0,
                    mode="bilinear",
                    align_corners=True,
                    antialias=False,
                )
                return a

        example_inputs = (torch.randn(2, 3, 4, 5),)
        # FIXME (T152380622)
        self.assertTrue(False)
        self.lower_and_test_with_partitioner(
            StaticResizeBilinear2DModule(), example_inputs
        )

    def test_xnnpack_backend_static_constant_pad(self):
        class StaticConstantPadModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                pad_6 = (1, 2, 3, 4, 5, 6)
                pad_4 = (1, 2, 3, 4)
                pad_2 = (1, 2)
                a = torch.nn.functional.pad(
                    input=x,
                    pad=pad_6,
                    mode="constant",
                    value=2.3,
                )
                b = torch.nn.functional.pad(
                    input=x,
                    pad=pad_4,
                    mode="constant",
                    value=1.3,
                )
                c = torch.nn.functional.pad(
                    input=x,
                    pad=pad_2,
                    mode="constant",
                    value=2.1,
                )
                d = torch.nn.functional.pad(
                    input=y,
                    pad=pad_6,
                    mode="constant",
                    value=2.7,
                )
                e = torch.nn.functional.pad(
                    input=y,
                    pad=pad_4,
                    mode="constant",
                    value=1.9,
                )
                f = torch.nn.functional.pad(
                    input=y,
                    pad=pad_2,
                    mode="constant",
                    value=3.1,
                )
                g = torch.nn.functional.pad(
                    input=z,
                    pad=pad_4,
                    mode="constant",
                    value=2.9,
                )
                h = torch.nn.functional.pad(
                    input=z,
                    pad=pad_2,
                    mode="constant",
                    value=1.2,
                )
                return (a, b, c, d, e, f, g, h)

        example_inputs = (
            torch.randn(size=(5, 4, 3, 2)),
            torch.randn(size=(5, 3, 2)),
            torch.randn(size=(4, 3)),
        )
        self.lower_module_and_test_output(StaticConstantPadModule(), example_inputs)
        self.lower_and_test_with_partitioner(StaticConstantPadModule(), example_inputs)

    def test_xnnpack_clamp(self):
        class Clamp(torch.nn.Module):
            def __init__(self, min_val, max_val):
                super().__init__()
                self.clamp = torch.clamp
                self.min_val = min_val
                self.max_val = max_val

            def forward(self, x):
                return self.clamp(x, min=self.min_val, max=self.max_val)

        model_inputs = (torch.randn(1, 4, 122, 122) * 2,)
        module = Clamp(-0.5, 0.5)
        self.lower_and_test_with_partitioner(module, model_inputs)

    def test_xnnpack_backend_maxpool2d(self):
        class MaxPool2dModule(torch.nn.Module):
            def __init__(
                self,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
            ):
                super().__init__()
                self.max_pool2d_module = torch.nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.max_pool2d_module(x)

        maxpool2d_module = MaxPool2dModule(3, 1, 0, 1)
        model_inputs = (torch.randn(4, 3, 24, 24),)

        self.lower_and_test_with_partitioner(maxpool2d_module, model_inputs)

    @unittest.expectedFailure
    def test_xnnpack_backend_maxpool2d_unsupported(self):
        class MaxPool2dModule(torch.nn.Module):
            def __init__(
                self,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
            ):
                super().__init__()
                self.max_pool2d_module = torch.nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    return_indices=True,
                )

            def forward(self, x):
                return self.max_pool2d_module(x)[1]

        maxpool2d_module = MaxPool2dModule(3, 1, 0, 1)
        model_inputs = (torch.randn(4, 3, 24, 24),)

        self.lower_and_test_with_partitioner(maxpool2d_module, model_inputs)

    @unittest.expectedFailure
    def test_xnnpack_backend_max_dim_vals(self):
        class MaxModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x):
                max_vals, _ = torch.max(x, dim=3, keepdim=True)
                return max_vals

        model_inputs = (torch.randn(16, 3, 12, 12),)
        max_dim_module = MaxModule()

        self.lower_and_test_with_partitioner(max_dim_module, model_inputs)

    def test_xnnpack_backend_max_dim(self):
        class MaxModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x):
                x = torch.add(x, x)
                max_values_1, max_indices_1 = torch.max(x, dim=2, keepdim=True)
                max_values_2, max_indices_2 = torch.max(x, dim=3, keepdim=True)
                return (max_values_1, max_indices_1, max_values_2, max_indices_2)

        model_inputs = (torch.randn(16, 3, 12, 12),)
        max_dim_module = MaxModule()

        self.lower_and_test_with_partitioner(max_dim_module, model_inputs)

    def test_xnnpack_backend_multiply(self):
        class MulModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.mul = torch.mul

            def forward(self, x, y):
                return self.mul(x, y)

        mul_module = MulModule()
        model_inputs = (
            torch.randn((1, 8)),
            torch.randn((8, 1)),
        )

        self.lower_and_test_with_partitioner(mul_module, model_inputs)

    def test_xnnpack_backend_sub(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = torch.sub

            def forward(self, x, y):
                return self.sub(x, y)

        module = Sub()
        M = torch.randn(2, 3)
        N = torch.randn(2, 3)
        model_inputs = (
            M,
            N,
        )
        self.lower_and_test_with_partitioner(module, model_inputs)

    def test_xnnpack_backend_floor(self):
        model_inputs = (torch.randn(1, 3, 3),)
        self.lower_and_test_with_partitioner(torch.floor, model_inputs)

    def test_xnnpack_backend_sqrt(self):
        model_inputs = (torch.randn(1, 3, 3).abs(),)
        self.lower_and_test_with_partitioner(torch.sqrt, model_inputs)

    def test_xnnpack_backend_ceil(self):
        model_inputs = (torch.randn(1, 3, 3),)
        self.lower_and_test_with_partitioner(torch.ceil, model_inputs)

    def test_xnnpack_backend_hardswish(self):
        # model_inputs = (torch.randn(1, 3, 3),)

        class HardswishModule(torch.nn.Module):
            def __init__(self):
                super(HardswishModule, self).__init__()
                self.hardswish_out_of_place = torch.nn.Hardswish()
                self.hardswish_in_place = torch.nn.Hardswish(inplace=True)
                self.hardswish_functional = torch.nn.functional.hardswish

            def forward(self, x):
                a = self.hardswish_out_of_place(x)
                a = self.hardswish_in_place(a)
                a = self.hardswish_functional(a)
                return a

        # TODO(T158969708)
        # self.lower_and_test_with_partitioner(HardswishModule(), model_inputs)

    # TODO(T158652796)
    @unittest.expectedFailure
    def test_xnnpack_backend_leaky_relu(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class LeakyReLUModule(torch.nn.Module):
            def __init__(self):
                super(LeakyReLUModule, self).__init__()
                self.leaky_relu_out_of_place = torch.nn.LeakyReLU(negative_slope=0.2)
                self.leaky_relu_in_place = torch.nn.LeakyReLU(
                    negative_slope=0.08, inplace=True
                )
                self.leaky_relu_functional_default = torch.nn.functional.leaky_relu

            def forward(self, x):
                a = self.leaky_relu_out_of_place(x)
                a = self.leaky_relu_in_place(a)
                a = self.leaky_relu_functional_default(a)
                return a

        self.lower_and_test_with_partitioner(LeakyReLUModule(), model_inputs)

    def test_xnnpack_channels_last_tagged_reshape_pass_output(self):
        op_sequences = OpSequencesAddConv2d(2, 2)
        op_sequences.eval()

        example_inputs = (torch.ones(1, 1, 6, 6),)

        self.lower_and_test_with_partitioner(op_sequences, example_inputs)

    def test_xnnpack_backend_conv2d_bn_hardtanh_mean_sequence(self):
        """
        This test makes sure that we can fuse batchnorm and hardtanh
        even with inserting copy nodes at some spots in the graph to change
        memory format
        """
        groups = 1
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 2
        out_channels = 1
        width = 8
        height = 8
        batches = 1
        example_inputs = (torch.randn(batches, in_channels, height, width),)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    dilation=dilation,
                    bias=True,
                )
                self.native_batchnorm = torch.nn.BatchNorm2d(out_channels)
                self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=6)

            def forward(self, x):
                x = self.conv(x)
                x = self.native_batchnorm(x)
                x = self.hardtanh(x)
                x = torch.mean(x, (-1, -2), keepdim=True)
                return x

        test_module = TestModule()
        test_module.eval()
        self.lower_and_test_with_partitioner(test_module, example_inputs)

    def test_xnnpack_backend_maximum(self):
        model_inputs_no_broadcast = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))
        model_inputs_broadcast = (torch.randn(2, 3, 4), torch.randn(2, 1, 4))

        self.lower_and_test_with_partitioner(torch.maximum, model_inputs_no_broadcast)
        self.lower_and_test_with_partitioner(torch.maximum, model_inputs_broadcast)

    def test_xnnpack_backend_negative(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class NegModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.neg(x)

        self.lower_and_test_with_partitioner(NegModule(), model_inputs)

    def test_xnnpack_backend_square(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class SquareModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.square(x)

        self.lower_and_test_with_partitioner(SquareModule(), model_inputs)

    @unittest.expectedFailure
    def test_xnnpack_backend_pow_unsupported(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class PowModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.pow(x, 4)

        self.lower_and_test_with_partitioner(PowModule(), model_inputs)

    def test_xnnpack_backend_elu(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.square(x)

        self.lower_and_test_with_partitioner(ELUModule(), model_inputs)

    def test_xnnpack_backend_avg_pool_2d(self):
        model_inputs = (torch.randn(1, 1, 10, 10),)

        class AvgPoolModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgPool = torch.nn.AvgPool2d(
                    kernel_size=(2, 2),
                    padding=(1, 1),
                    stride=(2, 2),
                    count_include_pad=False,
                )

            def forward(self, x):
                return self.avgPool(x)

        self.lower_and_test_with_partitioner(AvgPoolModule(), model_inputs)

    @unittest.expectedFailure
    def test_xnnpack_backend_avg_pool_2d_unsupported(self):
        model_inputs = (torch.randn(1, 1, 10, 10),)

        class AvgPoolModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgPool = torch.nn.AvgPool2d(
                    kernel_size=(2, 2),
                    padding=(1, 1),
                    stride=(2, 2),
                    count_include_pad=True,
                )

            def forward(self, x):
                return self.avgPool(x)

        self.lower_and_test_with_partitioner(AvgPoolModule(), model_inputs)

    @unittest.expectedFailure
    def test_xnnpack_backend_avg_pool_2d_unsupported2(self):
        model_inputs = (torch.randn(1, 1, 10, 10),)

        class AvgPoolModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgPool = torch.nn.AvgPool2d(
                    kernel_size=(2, 2),
                    padding=(1, 1),
                    stride=(2, 2),
                    count_include_pad=False,
                    ceil_mode=True,
                    divisor_override=4,
                )

            def forward(self, x):
                return self.avgPool(x)

        self.lower_and_test_with_partitioner(AvgPoolModule(), model_inputs)

    def test_xnnpack_backend_abs(self):
        model_inputs = (torch.randn(1, 3, 3),)

        class AbsModule(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)

        self.lower_and_test_with_partitioner(AbsModule(), model_inputs)

    # TODO(T158653285)
    @unittest.expectedFailure
    def test_xnnpack_backend_prelu(self):
        num_channels = 5
        model_inputs = (torch.randn(1, num_channels, 3, 2),)

        class PReLUModule(torch.nn.Module):
            def __init__(self):
                super(PReLUModule, self).__init__()
                self.prelu = torch.nn.PReLU()
                self.prelu_non_default = torch.nn.PReLU(
                    num_parameters=num_channels, init=0.2
                )

            def forward(self, x):
                a = self.prelu(x)
                a = self.prelu_non_default(a)
                return a

        self.lower_and_test_with_partitioner(PReLUModule(), model_inputs)

        # Should fail to be partitioned since constraint (input dim) is violated
        self.assertRaises(
            Exception,
            self.lower_and_test_with_partitioner,
            torch.nn.PReLU(),
            (torch.randn(1, 2),),
        )

    def test_xnnpack_backend_concatenate2(self):
        class Concat(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat((y, x), 0)

        self.lower_and_test_with_partitioner(
            Concat(), (torch.ones(4, 2, 3), torch.randn(1, 2, 3))
        )

    def test_xnnpack_backend_concatenate3(self):
        class Concat(torch.nn.Module):
            def forward(self, x, y):
                return torch.concat((y, y, x), 0)

        self.lower_and_test_with_partitioner(
            Concat(), (torch.ones(4, 2, 3), torch.randn(1, 2, 3))
        )

    def test_xnnpack_backend_concatenate4(self):
        class Concat(torch.nn.Module):
            def forward(self, x, y):
                return torch.concatenate((y, x, y, x), 2)

        self.lower_and_test_with_partitioner(
            Concat(), (torch.randn(1, 2, 3), torch.randn(1, 2, 5))
        )

    def test_xnnpack_backend_concatenate5(self):
        class Concat(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat((y, x, y, x, y), 2)

        self.assertRaises(
            Exception,
            self.lower_and_test_with_partitioner,
            Concat(),
            (torch.randn(1, 2, 3), torch.randn(1, 2, 5)),
        )

    def test_xnnpack_backend_concatenate_nhwc(self):
        class Concat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )

            def forward(self, x, y):
                x = self.conv(x)
                return torch.concatenate((y, x, y, x), 1)

        self.lower_and_test_with_partitioner(
            Concat(), (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 3, 3))
        )

    def test_xnnpack_backend_concatenate_nhwc2(self):
        class Concat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )

            def forward(self, x, y):
                x = self.conv(x)
                y = self.conv(y)
                return torch.concatenate((y, x, y, x), 3)

        self.lower_and_test_with_partitioner(
            Concat(), (torch.randn(1, 1, 3, 3), torch.randn(1, 1, 3, 3))
        )

    def test_xnnpack_backend_slice_copy(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[1:3, -2:, :-1]

        self.lower_and_test_with_partitioner(Slice(), (torch.randn(5, 5, 5),))

    def test_xnnpack_backend_slice_copy_stride_non_1(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[:3:-1, 2:, :3]

        self.assertRaises(
            Exception,
            self.lower_and_test_with_partitioner,
            Slice(),
            (torch.randn(5, 5, 5),),
        )

    def test_xnnpack_backend_slice_copy_dim_0(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[-1:3, 2:, 3:3]

        # Did not partition
        with self.assertRaisesRegex(IndexError, "list index out of range"):
            self.lower_module_and_test_output(
                Slice(), (torch.randn(5, 5, 5),), use_partitioner=True
            )

    def test_xnnpack_backend_slice_copy_memory_format(self):
        class ConvSlice(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding=1,
                    bias=False,
                )

            def forward(self, x):
                y = self.conv(x)
                return y[:, :, 2:3, -2:]

        self.lower_and_test_with_partitioner(ConvSlice(), (torch.randn(1, 1, 3, 3),))
