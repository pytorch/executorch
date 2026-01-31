# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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


class CortexMConvTranspose2D(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_transpose_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(*args, **kwargs, bias=False)
        self.conv_transpose.weight.data.fill_(1.0)

    def forward(self, x):
        return self.conv_transpose(x)


class CortexMConvTranspose2DBias(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_transpose_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(*args, **kwargs, bias=True)

    def forward(self, x):
        return self.conv_transpose(x)


# Test cases covering various configurations
test_cases = {
    # Basic test case
    "conv_transpose2d_basic": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, 3),
        example_inputs=(
            ramp_tensor(1, 5, (1, 2, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    # Stride variations
    "conv_transpose2d_stride_2": McuTestCase(
        model=CortexMConvTranspose2D(3, 6, kernel_size=3, stride=2),
        example_inputs=(
            ramp_tensor(0, 10, (1, 3, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_stride_asym": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, kernel_size=3, stride=(2, 3)),
        example_inputs=(
            ramp_tensor(-5, 5, (1, 2, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    # Padding variations
    "conv_transpose2d_padding_1": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, kernel_size=3, padding=1),
        example_inputs=(
            ramp_tensor(0, 20, (1, 2, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_padding_asym": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, kernel_size=3, padding=(2, 1)),
        example_inputs=(
            ramp_tensor(-10, 10, (1, 2, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    # Output padding variations (CRITICAL - unique to transpose conv)
    "conv_transpose2d_output_padding_1": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, kernel_size=3, stride=2, output_padding=1),
        example_inputs=(
            ramp_tensor(0, 15, (1, 2, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_output_padding_asym": McuTestCase(
        model=CortexMConvTranspose2D(
            3, 6, kernel_size=4, stride=2, output_padding=(1, 0)
        ),
        example_inputs=(
            ramp_tensor(5, 25, (1, 3, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    # Bias variations
    "conv_transpose2d_bias": McuTestCase(
        model=CortexMConvTranspose2DBias(4, 8, kernel_size=3),
        example_inputs=(
            ramp_tensor(-20, 20, (1, 4, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_bias_single_out": McuTestCase(
        model=CortexMConvTranspose2DBias(5, 1, kernel_size=3, stride=2),
        example_inputs=(
            ramp_tensor(0, 50, (1, 5, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    # Dilation variation
    "conv_transpose2d_dilation_2": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, kernel_size=3, dilation=2),
        example_inputs=(
            ramp_tensor(0, 30, (1, 2, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    # Grouped convolutions (not supported by CMSIS-NN)
    "conv_transpose2d_groups_2": McuTestCase(
        model=CortexMConvTranspose2D(4, 8, kernel_size=3, groups=2),
        example_inputs=(
            ramp_tensor(-15, 15, (1, 4, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_depthwise": McuTestCase(
        model=CortexMConvTranspose2D(4, 4, kernel_size=3, groups=4),
        example_inputs=(
            ramp_tensor(0, 40, (1, 4, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    # Kernel size variations
    "conv_transpose2d_kernel_1x1": McuTestCase(
        model=CortexMConvTranspose2D(3, 6, kernel_size=1, stride=2),
        example_inputs=(
            ramp_tensor(0, 12, (1, 3, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_kernel_asym": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, kernel_size=(2, 4)),
        example_inputs=(
            ramp_tensor(-8, 8, (1, 2, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_kernel_5x5": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, kernel_size=5, stride=2),
        example_inputs=(
            ramp_tensor(0, 25, (1, 2, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    # Channel variations
    "conv_transpose2d_single_channel_in": McuTestCase(
        model=CortexMConvTranspose2D(1, 8, kernel_size=3, stride=2),
        example_inputs=(
            ramp_tensor(0, 16, (1, 1, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_single_channel_out": McuTestCase(
        model=CortexMConvTranspose2D(8, 1, kernel_size=3, stride=2),
        example_inputs=(
            ramp_tensor(-40, 40, (1, 8, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_large_channels": McuTestCase(
        model=CortexMConvTranspose2D(16, 32, kernel_size=3),
        example_inputs=(
            ramp_tensor(-50, 50, (1, 16, 4, 4)).to(memory_format=torch.channels_last),
        ),
    ),
    # Input shape variations
    "conv_transpose2d_large_spatial": McuTestCase(
        model=CortexMConvTranspose2D(3, 6, kernel_size=3, stride=2),
        example_inputs=(
            ramp_tensor(-100, 100, (1, 3, 16, 16)).to(
                memory_format=torch.channels_last
            ),
        ),
    ),
    "conv_transpose2d_batch_2": McuTestCase(
        model=CortexMConvTranspose2D(2, 4, kernel_size=3),
        example_inputs=(
            ramp_tensor(0, 80, (2, 2, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_small_input": McuTestCase(
        model=CortexMConvTranspose2D(4, 8, kernel_size=3, stride=2),
        example_inputs=(
            ramp_tensor(0, 8, (1, 4, 2, 2)).to(memory_format=torch.channels_last),
        ),
    ),
    # Combined parameters
    "conv_transpose2d_complex": McuTestCase(
        model=CortexMConvTranspose2DBias(
            4, 8, kernel_size=4, stride=2, padding=1, output_padding=1
        ),
        example_inputs=(
            ramp_tensor(-30, 30, (1, 4, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv_transpose2d_all_params": McuTestCase(
        model=CortexMConvTranspose2DBias(
            3, 6, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=2
        ),
        example_inputs=(
            ramp_tensor(0, 60, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
}

xfails_dialect = {
    # Grouped convolutions not supported by CMSIS-NN - rejected during quantization
    "conv_transpose2d_groups_2": "Grouped transpose conv not supported by CMSIS-NN",
    "conv_transpose2d_depthwise": "Depthwise transpose conv not supported by CMSIS-NN",
    # output_padding not supported by CMSIS-NN - rejected during quantization
    "conv_transpose2d_output_padding_1": "output_padding not supported by CMSIS-NN",
    "conv_transpose2d_output_padding_asym": "output_padding not supported by CMSIS-NN",
    # dilation != 1 not supported by CMSIS-NN - rejected during quantization
    "conv_transpose2d_dilation_2": "dilation != 1 not supported by CMSIS-NN",
    # Combinations of unsupported features
    "conv_transpose2d_complex": "Uses output_padding which is not supported by CMSIS-NN",
    "conv_transpose2d_all_params": "Combines output_padding and dilation - both unsupported",
}


@parametrize("test_case", test_cases, xfails=xfails_dialect)
def test_dialect_conv_transpose2d(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


# Implementation xfails: empty because unsupported configurations are now
# rejected at AOT time by the quantizer filter, so they fall back to portable
# ops and work correctly. Only xfails_dialect needs to track these.
xfails_implementation = {}


@parametrize("test_case", test_cases, xfails=xfails_implementation)
def test_implementation_conv_transpose2d(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(qtol=2)
