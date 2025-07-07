# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

input_t = Tuple[torch.Tensor]  # Input x

exir_op = "executorch_exir_dialects_edge__ops_aten_convolution_default"

from executorch.backends.arm.test.ops.test_conv1d import Conv1d
from executorch.backends.arm.test.ops.test_conv2d import Conv2d


"""
The configuration when
  groups == in_channels and
  out_channels = K * in_channels
  where K is a positive integer
is termed in literature as depthwise convolution.
"""

dw_conv1d_3_1x3x14_gp3_st1 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=7,
    stride=1,
    groups=3,
    length=14,
    batches=1,
    padding=3,
)

dw_conv1d_2_1x6x4_gp6_st1 = Conv1d(
    in_channels=6,
    out_channels=12,
    kernel_size=2,
    stride=1,
    groups=6,
    padding=0,
    length=4,
    batches=1,
)

dw_conv2d_2x2_1x6x4x4_gp6_st1 = Conv2d(
    in_channels=6,
    out_channels=12,
    kernel_size=(2, 2),
    stride=(1, 1),
    groups=6,
    padding=0,
    width=4,
    height=4,
    batches=1,
)

dw_conv1d_3_1x3x256_gp3_st1 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=3,
    stride=1,
    groups=3,
    padding=0,
    length=256,
    batches=1,
)

dw_conv2d_3x3_1x3x256x256_gp3_st1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3),
    stride=(1, 1),
    groups=3,
    padding=0,
    width=256,
    height=256,
    batches=1,
)

dw_conv2d_3x3_1x4x256x256_gp4_st1 = Conv2d(
    in_channels=4,
    out_channels=8,
    kernel_size=(3, 3),
    stride=(1, 1),
    groups=4,
    padding=0,
    width=256,
    height=256,
    batches=1,
)

dw_conv2d_3x3_2x8x198x198_gp8_st3 = Conv2d(
    in_channels=8,
    out_channels=16,
    kernel_size=(3, 3),
    stride=3,
    groups=8,
    padding=0,
    width=198,
    height=198,
    batches=2,
)

dw_conv2d_3x3_1x4x256x256_gp4_nobias = Conv2d(
    in_channels=4,
    out_channels=8,
    kernel_size=(3, 3),
    stride=1,
    groups=4,
    bias=False,
    width=256,
    height=256,
    batches=1,
)

two_dw_conv1d = Conv1d(
    nbr_conv=2,
    length=64,
    in_channels=[4, 8],
    out_channels=[8, 24],
    kernel_size=[3, 3],
    stride=[1, 1],
    padding=[0, 0],
    groups=[4, 8],
    bias=[True, True],
    batches=1,
)

two_dw_conv2d = Conv2d(
    nbr_conv=2,
    width=64,
    height=64,
    in_channels=[4, 8],
    out_channels=[8, 24],
    kernel_size=[(3, 3), (3, 3)],
    stride=[1, 1],
    padding=[0, 0],
    groups=[4, 8],
    bias=[True, True],
    batches=2,
)

# Shenanigan to get a nicer output when test fails.
test_data_conv2d_MI = {
    "2x2_1x6x4x4_gp6_st1": lambda: dw_conv2d_2x2_1x6x4x4_gp6_st1,
    "3x3_1x3x256x256_gp3_st1": lambda: dw_conv2d_3x3_1x3x256x256_gp3_st1,
    "3x3_1x4x256x256_gp4_nobias": lambda: dw_conv2d_3x3_1x4x256x256_gp4_nobias,
    "3x3_1x4x256x256_gp4_st1": lambda: dw_conv2d_3x3_1x4x256x256_gp4_st1,
    "3x3_2x8x198x198_gp8_st3": lambda: dw_conv2d_3x3_2x8x198x198_gp8_st3,
    "two_dw_conv2d": lambda: two_dw_conv2d,
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_conv2d_BI = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_conv2d_MI.items()
    for q in [True, False]
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_conv2d_u85 = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in {
        "2x2_1x6x4x4_gp6_st1": lambda: dw_conv2d_2x2_1x6x4x4_gp6_st1,
        "3x3_1x3x256x256_gp3_st1": lambda: dw_conv2d_3x3_1x3x256x256_gp3_st1,
        "3x3_1x4x256x256_gp4_st1": lambda: dw_conv2d_3x3_1x4x256x256_gp4_st1,
        "3x3_1x4x256x256_gp4_nobias": lambda: dw_conv2d_3x3_1x4x256x256_gp4_nobias,
    }.items()
    for q in [True, False]
}

test_data_conv1d_MI = {
    "2_1x6x4_gp6_st1": lambda: dw_conv1d_2_1x6x4_gp6_st1,
    "two_dw_conv1d": lambda: two_dw_conv1d,
    "3_1x3x256_gp3_st1": lambda: dw_conv1d_3_1x3x256_gp3_st1,
    "3_1x3x14_gp3_st1": lambda: dw_conv1d_3_1x3x14_gp3_st1,
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_conv1d_BI = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_conv1d_MI.items()
    for q in [True, False]
}


@common.parametrize("test_data", test_data_conv1d_MI | test_data_conv2d_MI)
def test_depthwise_convolution_2d_tosa_MI(test_data: torch.nn.Module):
    pipeline = TosaPipelineMI[input_t](
        test_data(),
        test_data().get_inputs(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
@common.parametrize("test_data", test_data_conv1d_BI | test_data_conv2d_BI)
def test_depthwise_convolution_2d_tosa_BI(test_data):
    model, per_channel_quantization = test_data()
    pipeline = TosaPipelineBI[input_t](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


x_fails = {
    f"{k},per_channel_quant={q}": reason
    for k, reason in {
        "3x3_2x8x198x198_gp8_st3": "MLETORCH-517: Operators fail with batches > 1",
        "two_dw_conv2d": "MLETORCH-517: Operators fail with batches > 1",
    }.items()
    for q in [True, False]
}


@common.XfailIfNoCorstone300  # TODO: MLETORCH-516
@common.parametrize("test_data", test_data_conv2d_BI, x_fails)
def test_depthwise_convolution_2d_u55_BI(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineBI[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
        run_on_fvp=True,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.XfailIfNoCorstone300  # TODO: MLETORCH-516
@common.parametrize("test_data", test_data_conv1d_BI)
def test_depthwise_convolution_1d_u55_BI(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineBI[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
        run_on_fvp=True,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.XfailIfNoCorstone320  # TODO: MLETORCH-516
@common.parametrize("test_data", test_data_conv2d_BI, x_fails)
def test_depthwise_convolution_2d_u85_BI(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineBI[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
        run_on_fvp=True,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.XfailIfNoCorstone320  # TODO: MLETORCH-516
@common.parametrize("test_data", test_data_conv1d_BI, x_fails)
def test_depthwise_convolution_1d_u85_BI(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineBI[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
        run_on_fvp=True,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()
