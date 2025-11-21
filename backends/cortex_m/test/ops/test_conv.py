# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMConv1D(torch.nn.Module):
    ops_before_transforms = {}
    ops_after_transforms = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv1d(*args, **kwargs, bias=False)

    def forward(self, x):
        return self.conv(x)


class CortexMConv2D(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs, bias=False)
        self.conv.weight.data.fill_(1.0)

    def forward(self, x):
        return self.conv(x)


class CortexMConv2DBias(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs, bias=True)

    def forward(self, x):

        return self.conv(x)


class CortexMConv3D(torch.nn.Module):
    ops_before_transforms = {}

    ops_after_transforms = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv3d(*args, **kwargs, bias=False)
        self.conv.weight.data.fill_(2.0)

    def forward(self, x):
        return self.conv(x)


class CortexMConv2Dx3(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 3,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(16, 8, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CortexMConv2DReLU(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
test_cases = {
    "conv2d": McuTestCase(
        model=CortexMConv2D(2, 4, 3),
        example_inputs=(
            ramp_tensor(1, 5, (1, 2, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_stride": McuTestCase(
        model=CortexMConv2D(3, 4, (1, 2), stride=2),
        example_inputs=(
            ramp_tensor(-100, 10, (3, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_padding": McuTestCase(
        model=CortexMConv2D(3, 2, 3, padding=(4, 1)),
        example_inputs=(
            ramp_tensor(0, 1, (2, 3, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_dilation": McuTestCase(
        model=CortexMConv2D(1, 4, 3, dilation=(2, 2)),
        example_inputs=(
            ramp_tensor(0, 10, (3, 1, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_groups": McuTestCase(
        model=CortexMConv2D(4, 4, 1, groups=2),
        example_inputs=(
            ramp_tensor(0, 10, (1, 4, 1, 1)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_bias_ch_out_1": McuTestCase(
        model=CortexMConv2DBias(5, 1, 1),
        example_inputs=(
            ramp_tensor(0, 10, (2, 5, 3, 3)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_bias_ch_out_4": McuTestCase(
        model=CortexMConv2DBias(5, 4, (1, 2)),
        example_inputs=(
            ramp_tensor(-3, 3, (2, 5, 10, 10)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_nchw": McuTestCase(
        model=CortexMConv2D(5, 5, 1),
        example_inputs=(ramp_tensor(0, 10, (1, 5, 8, 8)),),
    ),
    "conv1d": McuTestCase(
        model=CortexMConv1D(1, 1, 1),
        example_inputs=(ramp_tensor(0, 10, (1, 3, 2)),),
    ),
    "conv3d": McuTestCase(
        model=CortexMConv3D(1, 1, 1),
        example_inputs=(
            ramp_tensor(-1000, 1000, (2, 1, 3, 3, 3)).to(
                memory_format=torch.channels_last_3d
            ),
        ),
    ),
    "conv2d_x3": McuTestCase(
        model=CortexMConv2Dx3(),
        example_inputs=(
            ramp_tensor(0, 10, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_relu": McuTestCase(
        model=CortexMConv2DReLU(),
        example_inputs=(
            ramp_tensor(-5, 5, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
}


xfails_dialect = {
    "conv2d_dilation": "NotImplementedError: 'slow_conv_dilated<>' not implemented for 'Int'",
    "conv1d": "Currently not supported.",
    "conv2d_nchw": "Currently not supported.",
    "conv3d": "Currently not supported.",
    "conv2d_relu": "Currently not supported.",
}


@parametrize("test_case", test_cases, xfails=xfails_dialect)
def test_dialect_conv2d(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


xfails_implementation = {
    "conv1d": "Currently not supported.",
    "conv2d_nchw": "Currently not supported.",
    "conv3d": "Currently not supported.",
    "conv2d_relu": "Currently not supported.",
}


@parametrize("test_case", test_cases, xfails=xfails_implementation)
def test_implementation_conv2d(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=1)
