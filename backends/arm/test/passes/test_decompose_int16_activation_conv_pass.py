# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for DecomposeConvWithInt16ActivationPass.

This pass decomposes convolution with int16 activation and bias into:
- A convolution without bias
- A rescale to int32
- An add with the reshaped bias
- A rescale back to the output dtype
"""

from typing import Tuple

import torch
from executorch.backends.arm._passes import DecomposeConvWithInt16ActivationPass
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor]


class Conv2dWithBias(torch.nn.Module):
    """Conv2d with bias - should be decomposed for INT16 activations."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2dWithoutBias(torch.nn.Module):
    """Conv2d without bias - should NOT be decomposed."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2dMultipleConvs(torch.nn.Module):
    """Multiple Conv2d layers with bias - all should be decomposed for INT16."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def test_decompose_int16_conv_pass_fp32_no_decomposition() -> None:
    """
    Test that DecomposeConvWithInt16ActivationPass does NOT decompose
    convolution when using FP32 (no quantization).
    """
    module = Conv2dWithBias()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_add_Tensor",
            "executorch_exir_dialects_backend__ops_tosa_RESCALE_default",
        ],
        pass_list=[DecomposeConvWithInt16ActivationPass],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


aten_op = "torch.ops.aten.conv2d.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_convolution_default"


def test_conv2d_int16_e2e_tosa_single_conv() -> None:
    """
    End-to-end test for conv2d with INT16 quantization using TOSA pipeline.
    This validates the full lowering path including the decomposition pass
    for a single convolution with bias.
    """
    module = Conv2dWithBias()
    pipeline = TosaPipelineINT[input_t](
        module,
        module.get_inputs(),
        aten_op,
        exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


def test_conv2d_int16_e2e_tosa_multiple_convs() -> None:
    """
    End-to-end test for conv2d with INT16 quantization using TOSA pipeline.
    This validates the full lowering path including the decomposition pass
    for multiple convolutions with bias.
    """
    module = Conv2dMultipleConvs()
    pipeline = TosaPipelineINT[input_t](
        module,
        module.get_inputs(),
        aten_op,
        exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


def test_conv2d_int16_e2e_tosa_without_bias() -> None:
    """
    End-to-end test for conv2d without bias with INT16 quantization.
    This validates that convolutions without bias don't get decomposed.
    """
    module = Conv2dWithoutBias()
    pipeline = TosaPipelineINT[input_t](
        module,
        module.get_inputs(),
        aten_op,
        exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


def test_conv2d_int8_e2e_tosa() -> None:
    """
    End-to-end test for conv2d with INT8 quantization using TOSA pipeline.
    This validates that INT8 activations don't trigger the decomposition.
    """
    module = Conv2dWithBias()
    pipeline = TosaPipelineINT[input_t](
        module,
        module.get_inputs(),
        aten_op,
        exir_op,
    )
    pipeline.run()
