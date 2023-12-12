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

from executorch.exir.dialects._ops import ops as exir_ops

from torch.ao.quantization.observer import (
    per_channel_weight_observer_range_neg_127_to_127,
    weight_observer_range_neg_127_to_127,
)


class TestXNNPACKQuantized(TestXNNPACK):
    def test_xnnpack_q_per_tensor(self):
        def just_quant(x):
            return exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                x, 0.12345, 0, -127, 127, torch.int8
            )

        sample_input = (torch.randn(1, 1, 4, 4),)
        self.lower_module_and_test_output(just_quant, sample_input)

    def test_xnnpack_dq_per_tensor(self):
        def just_quant(x):
            return exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                x, 0.12345, 0, -127, 127, torch.int8
            )

        sample_input = (
            (
                torch.randint(low=-127, high=127, size=(1, 1, 4, 4)).type(
                    dtype=torch.int8
                )
            ),
        )
        self.lower_module_and_test_output(just_quant, sample_input)

    def test_xnnpack_qmax_pool_2d(self):
        class maxpool(torch.nn.Module):
            def __init__(self, maxpool_params):
                super().__init__()
                self.max = torch.nn.MaxPool2d(*maxpool_params)

            def forward(self, x):
                return self.max(x)

        for maxpool_params in [(4,), (4, 2), (4, 2, 2)]:
            example_input = (torch.ones(1, 2, 8, 8),)
            self.quantize_and_test_model(maxpool(maxpool_params), example_input)

    def test_xnnpack_qadd(self):
        class Add(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        example_inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        self.quantize_and_test_model(Add(), example_inputs)

    def test_xnnpack_qadd2(self):
        class Add(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        example_inputs = (torch.randn(1, 1, 4, 4),)
        self.quantize_and_test_model(Add(), example_inputs)

    def test_xnnpack_qsub(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x - y

        example_inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        self.quantize_and_test_model(Sub(), example_inputs)

    def test_xnnpack_qsub2(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x - x

        example_inputs = (torch.randn(1, 1, 4, 4),)
        self.quantize_and_test_model(Sub(), example_inputs)

    def test_xnnpack_qsub3(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x - y

        example_inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1))
        self.quantize_and_test_model(Sub(), example_inputs)

    def test_xnnpack_qsub_relu(self):
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x - y
                return torch.nn.functional.relu(z)

        example_inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        self.quantize_and_test_model(Sub(), example_inputs)

    def test_xnnpack_qmul(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x * y

        example_inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 1))
        self.quantize_and_test_model(Mul(), example_inputs)

    def test_xnnpack_qmul2(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x * x

        example_inputs = (torch.randn(1, 1, 4, 4),)
        self.quantize_and_test_model(Mul(), example_inputs)

    def test_xnnpack_qmul_functional(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.mul(x, y) * torch.functional.torch.mul(x, y)

        example_inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        self.quantize_and_test_model(Mul(), example_inputs)

    def test_xnnpack_qmul_relu(self):
        class Mul(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x * y
                return torch.nn.functional.relu(z)

        example_inputs = (torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        self.quantize_and_test_model(Mul(), example_inputs)

    def test_xnnpack_qmean(self):
        class Mean(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.mean((-2, -1), keepdim=True)

        example_inputs = (torch.randn(1, 1, 4, 4),)
        self.quantize_and_test_model(Mean(), example_inputs)

    def test_xnnpack_qhardtanh(self):
        example_inputs = (torch.randn(1, 1, 1),)
        self.quantize_and_test_model(torch.nn.Hardtanh(), example_inputs)

    def test_xnnpack_leaky_relu(self):
        example_inputs = (torch.randn(1, 3, 3),)

        class LeakyReLUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaky_relu_out_of_place = torch.nn.LeakyReLU(negative_slope=0.2)

            def forward(self, x):
                return self.leaky_relu_out_of_place(x)

        self.quantize_and_test_model(LeakyReLUModule(), example_inputs)

    def test_xnnpack_leaky_relu2(self):
        example_inputs = (torch.randn(1, 3, 3),)

        class LeakyReLUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaky_relu_in_place = torch.nn.LeakyReLU(
                    negative_slope=0.08, inplace=True
                )

            def forward(self, x):
                return self.leaky_relu_in_place(x)

        self.quantize_and_test_model(LeakyReLUModule(), example_inputs)

    def test_xnnpack_leaky_relu3(self):
        example_inputs = (torch.randn(1, 3, 3),)

        class LeakyReLUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaky_relu_functional_default = torch.nn.functional.leaky_relu

            def forward(self, x):
                return self.leaky_relu_functional_default(x)

        self.quantize_and_test_model(LeakyReLUModule(), example_inputs)

    def test_xnnpack_qlinear(self):
        in_size = 1
        input_size = 3
        output_size = 4
        linear = torch.nn.Linear(input_size, output_size)

        linear.weight = torch.nn.Parameter(
            torch.randn(output_size, input_size, dtype=torch.float)
        )
        linear.bias = torch.nn.Parameter(torch.ones(output_size, dtype=torch.float))

        example_inputs = (torch.randn(in_size, input_size, dtype=torch.float),)
        self.quantize_and_test_model(
            linear,
            example_inputs,
        )

    def test_xnnpack_multiuse_qconv(self):
        class ReusedInput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                return self.conv1(x) + self.conv2(x)

        example_inputs = (torch.randn(1, 1, 4, 4),)
        self.quantize_and_test_model(ReusedInput(), example_inputs)

    def test_xnnpack_qconv_bn(self):
        class ModelConvBN(torch.nn.Module):
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

        model = ModelConvBN().eval()
        example_inputs = (torch.randn(2, 2, 4, 4),)
        self.quantize_and_test_model(model, example_inputs)

    def get_qconv2d(self, has_bias):
        groups = 1
        stride = [2, 2]
        padding = [1, 1]
        dilation = [1, 1]
        in_channels = 2
        out_channels = 1
        width = 8
        height = 8
        batches = 1
        conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=has_bias,
        )
        example_inputs = (
            torch.randn(batches, in_channels, height, width, dtype=torch.float),
        )
        return (conv, example_inputs)

    def test_xnnpack_qconv2d(self):
        for has_bias in (True, False):
            conv, example_inputs = self.get_qconv2d(has_bias)
            self.quantize_and_test_model(
                conv,
                example_inputs,
            )

    def test_xnnpack_qconv2d_per_channel(self):
        for has_bias in (True,):
            conv, example_inputs = self.get_qconv2d(has_bias)
            self.quantize_and_test_model(
                conv,
                example_inputs,
                per_channel_quant=True,
            )

    def test_xnnpack_qconv_relu(self):
        class ModelConvReLU(torch.nn.Module):
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

        model = ModelConvReLU().eval()
        example_inputs = (torch.randn(2, 2, 4, 4),)
        self.quantize_and_test_model(model, example_inputs)

    def test_xnnpack_dw_qconv_relu(self):
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

        model = ModelConvReLU().eval()
        example_inputs = (torch.randn(batches, in_channels, height, width) * 11,)
        for per_channel_quant in (True, False):
            self.quantize_and_test_model(
                model,
                example_inputs,
                per_channel_quant=per_channel_quant,
            )

    def test_xnnpack_qconv_relu_sequence(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1, 64, 1),
            torch.nn.ReLU(),
        )
        model = model.eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self.quantize_and_test_model(model, example_inputs)

    def test_xnnpack_qadd_relu(self):
        class addrelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = x + x
                return self.relu(y)

        model = addrelu().eval()
        example_inputs = (
            -torch.ones(
                1,
                1,
                20,
                20,
            ),
        )
        self.quantize_and_test_model(model, example_inputs)

    def test_xnnpack_qadd_relu_seq(self):
        class addrelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x, z):
                y = x + z
                y = self.relu(y)
                y = y + y
                y = self.relu(y)
                return y

        model = addrelu().eval()
        example_inputs = (
            torch.randn(
                1,
                1,
                20,
                20,
            ),
            torch.randn(
                1,
                1,
                20,
                20,
            ),
        )
        self.quantize_and_test_model(model, example_inputs)

    def test_xnnpack_qclamp(self):
        class Clamp(torch.nn.Module):
            def __init__(self, min_val, max_val):
                super().__init__()
                self.min_val = min_val
                self.max_val = max_val

            def forward(self, x):
                return torch.clamp(x + x, min=self.min_val, max=self.max_val)

        model_inputs = (torch.randn(1, 4, 122, 122),)
        module = Clamp(-1, 1)
        self.quantize_and_test_model(module, model_inputs)

    def test_xnnpack_qpermute(self):
        class Perm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nchw_to_nhwc = [0, 2, 3, 1]

            def forward(self, x):
                return torch.permute(x, self.nchw_to_nhwc)

        self.quantize_and_test_model(Perm(), (torch.randn(1, 2, 4, 5),))

    def test_xnnpack_qpermute_copy(self):
        class Perm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nchw_to_nhwc = [0, 2, 3, 1]

            def forward(self, x):
                return torch.permute_copy(x, self.nchw_to_nhwc)

        self.quantize_and_test_model(Perm(), (torch.randn(1, 2, 4, 5),))

    def test_xnnpack_qconstant_pad(self):
        class StaticConstantPadModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cp = torch.nn.ConstantPad2d([1, 2, 3, 4], 2.3)

            def forward(self, x):
                a = self.cp(x)
                return a

        example_inputs = (torch.randn(5, 4, 3, 2),)
        self.quantize_and_test_model(StaticConstantPadModule(), example_inputs)

    def test_xnnpack_qconstant_pad2(self):
        class StaticConstantPadModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                a = torch.nn.functional.pad(
                    x, pad=(1, 2, 3, 4, 5, 6), mode="constant", value=1.3
                )
                return a

        example_inputs = (torch.randn(5, 4, 3, 2),)
        self.quantize_and_test_model(StaticConstantPadModule(), example_inputs)

    # TODO(T158652796)
    @unittest.expectedFailure
    def test_xnnpack_qelu(self):
        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.elu = torch.nn.ELU(alpha=0.5)

            def forward(self, x):
                return self.elu(x)

        example_inputs = (torch.randn(1, 3, 4, 4),)
        self.quantize_and_test_model(ELUModule(), example_inputs)

    # TODO(T158652796)
    @unittest.expectedFailure
    def test_xnnpack_qelu2(self):
        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.elu(x, alpha=1.2)

        example_inputs = (torch.randn(1, 3, 4, 4),)
        self.quantize_and_test_model(ELUModule(), example_inputs)

    def test_xnnpack_qcat2(self):
        class CatModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat((x, y), axis=2)

        example_inputs = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        self.quantize_and_test_model(CatModule(), example_inputs)

    def test_xnnpack_qcat3(self):
        class CatModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.concat((y, y, x), axis=2)

        example_inputs = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        self.quantize_and_test_model(CatModule(), example_inputs)

    def test_xnnpack_qcat4(self):
        class CatModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.concatenate((y, y, x, x), axis=2)

        example_inputs = (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 2))
        self.quantize_and_test_model(CatModule(), example_inputs)

    def test_xnnpack_qslice(self):
        class ModelConvSlice(torch.nn.Module):
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

            def forward(self, x):
                y = self.conv1(x)
                return y[0:1, 0:1]

        model = ModelConvSlice().eval()
        example_inputs = (torch.randn(2, 2, 4, 4),)

        self.quantize_and_test_model_with_quantizer(
            model,
            example_inputs,
        )

    @unittest.skip("Dynamic Per Tensor Quantization is not supported yet")
    def test_xnnpack_dqlinear_mm_per_tensor(self):
        self._test_xnnpack_dqlinear(
            weight_qconfig=weight_observer_range_neg_127_to_127, use_bias=False
        )

    @unittest.skip("Dynamic Per Tensor Quantization is not supported yet")
    def test_xnnpack_dqlinear_addmm_per_tensor(self):
        self._test_xnnpack_dqlinear(
            weight_qconfig=weight_observer_range_neg_127_to_127, use_bias=True
        )

    def test_xnnpack_dqlinear_mm_per_channel(self):
        self._test_xnnpack_dqlinear(
            weight_qconfig=per_channel_weight_observer_range_neg_127_to_127,
            use_bias=False,
        )

    def test_xnnpack_dqlinear_addmm_per_channel(self):
        self._test_xnnpack_dqlinear(
            weight_qconfig=per_channel_weight_observer_range_neg_127_to_127,
            use_bias=True,
        )

    @unittest.skip("Dynamic Per Tensor Quantization is not supported yet")
    def test_xnnpack_dqlinear_partitioner_mm_per_tensor(self):
        self._test_xnnpack_dqlinear_with_partitioner(
            weight_qconfig=weight_observer_range_neg_127_to_127, use_bias=False
        )

    @unittest.skip("Dynamic Per Tensor Quantization is not supported yet")
    def test_xnnpack_dqlinear_partitioner_addmm_per_tensor(self):
        self._test_xnnpack_dqlinear_with_partitioner(
            weight_qconfig=weight_observer_range_neg_127_to_127, use_bias=True
        )

    def test_xnnpack_dqlinear_partitioner_mm_per_channel(self):
        self._test_xnnpack_dqlinear_with_partitioner(
            weight_qconfig=per_channel_weight_observer_range_neg_127_to_127,
            use_bias=False,
        )

    def test_xnnpack_dqlinear_partitioner_addmm_per_channel(self):
        self._test_xnnpack_dqlinear_with_partitioner(
            weight_qconfig=per_channel_weight_observer_range_neg_127_to_127,
            use_bias=True,
        )

    def test_xnnpack_multi_dqlinear_with_partitioner_parallel(self):
        use_bias = True

        in_size = 1
        input_size = 4
        output_size = 5

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(output_size, input_size)
                )
                self.linear1_bias = (
                    torch.nn.Parameter(torch.rand(output_size)) if use_bias else None
                )

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(output_size, input_size)
                )
                self.linear2_bias = (
                    torch.nn.Parameter(torch.rand(output_size)) if use_bias else None
                )

            def forward(self, x, y):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    y, self.linear2_weight, self.linear2_bias
                )
                return (a, b)

        example_inputs = (
            torch.rand(in_size, input_size, dtype=torch.float),
            torch.rand(in_size, input_size, dtype=torch.float),
        )

        self._test_xnnpack_custom_dqlinear_with_partitioner_only(
            LinearModule, example_inputs
        )

    def test_xnnpack_multi_dqlinear_with_partitioner_sequential(self):
        use_bias = True

        in_size = 1
        input_size = 4
        intermediate_size = 5
        output_size = 3

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear1_bias = (
                    torch.nn.Parameter(torch.rand(intermediate_size))
                    if use_bias
                    else None
                )

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(output_size, intermediate_size)
                )
                self.linear2_bias = (
                    torch.nn.Parameter(torch.rand(output_size)) if use_bias else None
                )

            def forward(self, x):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    a, self.linear2_weight, self.linear2_bias
                )
                return b

        example_inputs = (torch.rand(in_size, input_size, dtype=torch.float),)

        self._test_xnnpack_custom_dqlinear_with_partitioner_only(
            LinearModule, example_inputs
        )

    def test_xnnpack_multi_dqlinear_with_partitioner_parallel_and_sequential(self):
        use_bias = True

        in_size = 1
        input_size = 4
        intermediate_size = 5
        output_size = 3

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear1_bias = (
                    torch.nn.Parameter(torch.rand(intermediate_size))
                    if use_bias
                    else None
                )

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear2_bias = (
                    torch.nn.Parameter(torch.rand(intermediate_size))
                    if use_bias
                    else None
                )

                self.linear3_weight = torch.nn.Parameter(
                    torch.rand(output_size, intermediate_size)
                )
                self.linear3_bias = (
                    torch.nn.Parameter(torch.rand(output_size)) if use_bias else None
                )

            def forward(self, x, y):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    y, self.linear2_weight, self.linear2_bias
                )
                c = torch.nn.functional.linear(
                    b, self.linear3_weight, self.linear3_bias
                )
                return (a, c)

        example_inputs = (
            torch.rand(in_size, input_size, dtype=torch.float),
            torch.rand(in_size, input_size, dtype=torch.float),
        )

        self._test_xnnpack_custom_dqlinear_with_partitioner_only(
            LinearModule, example_inputs
        )
