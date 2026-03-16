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


class CortexMLinearReLU(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class CortexMLinearHardtanh(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, min_val=-0.25, max_val=0.75):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.act = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return self.act(self.linear(x))


class CortexMLinearReLU6(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, in_features=8, out_features=8):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.relu6 = torch.nn.ReLU6()

    def forward(self, x):
        return self.relu6(self.linear(x))


class CortexMLinearReLUInplace(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, in_features=8, out_features=8):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.linear(x))


class CortexMLinearHardtanhInplace(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        self.act = torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=True)

    def forward(self, x):
        return self.act(self.linear(x))


class CortexMLinearHardsigmoid(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardsigmoid_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, in_features=6, out_features=6):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.act = torch.nn.Hardsigmoid()

    def forward(self, x):
        return self.act(self.linear(x))


class CortexMLinearHardswish(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_clamp_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardswish_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, in_features=8, out_features=8):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.act = torch.nn.Hardswish()

    def forward(self, x):
        return self.act(self.linear(x))


class CortexMConv2DReLU(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class CortexMConv2DReLU6(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=2, padding=1, bias=False)
        self.relu6 = torch.nn.ReLU6()

    def forward(self, x):
        return self.relu6(self.conv(x))


class CortexMConv2DHardtanh(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, min_val=-2.0, max_val=2.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1, bias=True)
        self.act = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward(self, x):
        return self.act(self.conv(x))


class CortexMConv2DHardswish(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_clamp_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardswish_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_minimum_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, padding=0, bias=False)
        self.act = torch.nn.Hardswish()
        self.conv.weight.data.fill_(0.5)

    def forward(self, x):
        return self.act(self.conv(x))


class CortexMConv2DReLUInplace(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class CortexMConv2DHardtanhInplace(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, min_val=-0.5, max_val=0.5):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False)
        self.act = torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=True)
        torch.nn.init.ones_(self.conv.weight)

    def forward(self, x):
        return self.act(self.conv(x))


class CortexMConv2DHardsigmoid(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_hardsigmoid_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, bias=False)
        self.act = torch.nn.Hardsigmoid(inplace=True)
        self.conv.weight.data.fill_(0.5)

    def forward(self, x):
        return self.act(self.conv(x))


class CortexMConv2DClampInplace(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_aten_clamp_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, bias=False)
        self.conv.weight.data.fill_(0.5)

    def forward(self, x):
        return torch.clamp_(self.conv(x), min=0.0, max=None)


class CortexMLinearClamp(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_aten_clamp_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return torch.clamp(self.linear(x), min=None, max=6.0)


test_cases = {
    # Linear + activation tests with various data ranges
    "linear_relu_small_range": McuTestCase(
        model=CortexMLinearReLU(),
        example_inputs=(ramp_tensor(-10, 10, (1, 4)),),
    ),
    "linear_relu_large_range": McuTestCase(
        model=CortexMLinearReLU(in_features=16, out_features=16),
        example_inputs=(ramp_tensor(-100, 100, (2, 16)),),
    ),
    "linear_relu_negative": McuTestCase(
        model=CortexMLinearReLU(in_features=8, out_features=8),
        example_inputs=(ramp_tensor(-50, 0, (1, 8)),),
    ),
    "linear_relu6": McuTestCase(
        model=CortexMLinearReLU6(),
        example_inputs=(ramp_tensor(-2, 10, (1, 8)),),
    ),
    "linear_relu_inplace": McuTestCase(
        model=CortexMLinearReLUInplace(),
        example_inputs=(ramp_tensor(-5, 5, (2, 8)),),
    ),
    "linear_hardtanh_symmetric": McuTestCase(
        model=CortexMLinearHardtanh(min_val=-0.5, max_val=0.5),
        example_inputs=(ramp_tensor(-1, 1, (2, 1, 4)),),
    ),
    "linear_hardtanh_asymmetric": McuTestCase(
        model=CortexMLinearHardtanh(min_val=-1.5, max_val=0.25),
        example_inputs=(ramp_tensor(-2, 1, (1, 4)),),
    ),
    "linear_hardtanh_large_range": McuTestCase(
        model=CortexMLinearHardtanh(min_val=-10.0, max_val=10.0),
        example_inputs=(ramp_tensor(-20, 20, (2, 4)),),
    ),
    "linear_hardtanh_inplace": McuTestCase(
        model=CortexMLinearHardtanhInplace(min_val=-0.75, max_val=0.75),
        example_inputs=(ramp_tensor(-2, 2, (1, 8)),),
    ),
    # Convolution + activation tests with various configurations
    "conv2d_relu_small_kernel": McuTestCase(
        model=CortexMConv2DReLU(),
        example_inputs=(
            ramp_tensor(-5, 5, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_relu_large_range": McuTestCase(
        model=CortexMConv2DReLU(),
        example_inputs=(
            ramp_tensor(-50, 50, (2, 4, 16, 16)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_relu6_stride": McuTestCase(
        model=CortexMConv2DReLU6(),
        example_inputs=(
            ramp_tensor(-10, 20, (1, 3, 12, 12)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_relu_inplace": McuTestCase(
        model=CortexMConv2DReLUInplace(),
        example_inputs=(
            ramp_tensor(-3, 3, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_hardtanh_narrow": McuTestCase(
        model=CortexMConv2DHardtanh(min_val=-0.5, max_val=0.5),
        example_inputs=(
            ramp_tensor(-2, 2, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_hardtanh_wide": McuTestCase(
        model=CortexMConv2DHardtanh(min_val=-5.0, max_val=5.0),
        example_inputs=(
            ramp_tensor(-10, 10, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_hardtanh_inplace": McuTestCase(
        model=CortexMConv2DHardtanhInplace(min_val=-10.0, max_val=10.0),
        example_inputs=(
            ramp_tensor(-15, 15, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "linear_hardsigmoid": McuTestCase(
        model=CortexMLinearHardsigmoid(in_features=6, out_features=4),
        example_inputs=(ramp_tensor(-8, 8, (2, 6)),),
    ),
    "linear_hardswish": McuTestCase(
        model=CortexMLinearHardswish(in_features=12, out_features=6),
        example_inputs=(ramp_tensor(-2, 0, (1, 12)),),
    ),
    "conv2d_hardsigmoid_inplace": McuTestCase(
        model=CortexMConv2DHardsigmoid(),
        example_inputs=(
            ramp_tensor(-4, 4, (1, 2, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_hardswish": McuTestCase(
        model=CortexMConv2DHardswish(),
        example_inputs=(
            ramp_tensor(-3, 0, (1, 2, 1, 100)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_clamp_inplace": McuTestCase(
        model=CortexMConv2DClampInplace(),
        example_inputs=(
            ramp_tensor(-4, 4, (1, 2, 1, 10)).to(memory_format=torch.channels_last),
        ),
    ),
    "linear_clamp": McuTestCase(
        model=CortexMLinearClamp(in_features=4, out_features=3),
        example_inputs=(ramp_tensor(-10, 10, (1, 4)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_activation(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_activation(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=1)
