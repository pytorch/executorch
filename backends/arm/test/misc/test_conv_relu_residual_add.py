# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
import torch.nn as nn
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)


# Model with Conv1D - ReLU sequence and a residual add.
# Testing the annotation of Conv1D-ReLU(to be fused) and annotation of add.
# ReLU outputs positive numbers and linear outputs positive and negative numbers, so they
# should have different quantisation parameters. If the ReLU gets wrong quantisation parameters(e.g. qmin!=zp)
# because of a shared observer of a following operators(e.g. add),  the Conv1D-ReLU sequence is not fused
# and is left in FP32. As a result, the test fails.
class AddDifferentRanges(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = torch.nn.ReLU()
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        # Permute: (N, T, C) -> (N, C, T)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        out = x + self.linear(x)
        return out


input_t = Tuple[torch.Tensor]
model = AddDifferentRanges(in_channels=3, out_channels=16, kernel_size=3, input_dim=10)
model_inputs = (torch.randn(1, 10, 3),)
quant_test_data = {
    "per_channel_quantization=true": True,
    "per_channel_quantization=false": False,
}


def test_conv_relu_residual_add_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("per_channel_quantization", quant_test_data)
def test_conv_relu_residual_add_tosa_INT(per_channel_quantization):
    pipeline = TosaPipelineINT[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        qtol=0,
    )
    pipeline.run()


# TODO: Xfail until the Ethos-U Vela compiler ships commit
# 642f7517d3a6bd053032e1942822f6e38ccd546f. That patch fixes the bug that
# causes this test to fail.
@pytest.mark.xfail(
    reason=("Blocked by Vela commit 642f7517d3a6bd053032e1942822f6e38ccd546f"),
    strict=True,
)
@pytest.mark.slow
@common.XfailIfNoCorstone300
@common.parametrize("per_channel_quantization", quant_test_data)
def test_conv_relu_residual_add_u55_INT(per_channel_quantization):
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model_inputs,
        [],
        [],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        qtol=0,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone320
@common.parametrize("per_channel_quantization", quant_test_data)
def test_conv_relu_residual_add_u85_INT(per_channel_quantization):
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model_inputs,
        [],
        [],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        qtol=0,
    )
    pipeline.run()
