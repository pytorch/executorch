# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a8w4_quantization_config,
)
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
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

dw_conv1d_3_1x3x32_gp3_st1 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=3,
    stride=1,
    groups=3,
    padding=0,
    length=32,
    batches=1,
)

dw_conv2d_3x3_1x3x24x24_gp3_st1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3),
    stride=(1, 1),
    groups=3,
    padding=0,
    width=24,
    height=24,
    batches=1,
)

dw_conv2d_3x3_1x4x24x24_gp4_st1 = Conv2d(
    in_channels=4,
    out_channels=8,
    kernel_size=(3, 3),
    stride=(1, 1),
    groups=4,
    padding=0,
    width=24,
    height=24,
    batches=1,
)

dw_conv2d_3x3_2x8x27x27_gp8_st3 = Conv2d(
    in_channels=8,
    out_channels=16,
    kernel_size=(3, 3),
    stride=3,
    groups=8,
    padding=0,
    width=27,
    height=27,
    batches=2,
)

dw_conv2d_3x3_1x4x24x24_gp4_nobias = Conv2d(
    in_channels=4,
    out_channels=8,
    kernel_size=(3, 3),
    stride=1,
    groups=4,
    bias=False,
    width=24,
    height=24,
    batches=1,
)

two_dw_conv1d = Conv1d(
    nbr_conv=2,
    length=16,
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
    width=24,
    height=24,
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
test_data_conv2d_FP = {
    "2x2_1x6x4x4_gp6_st1": lambda: dw_conv2d_2x2_1x6x4x4_gp6_st1,
    "3x3_1x3x24x24_gp3_st1": lambda: dw_conv2d_3x3_1x3x24x24_gp3_st1,
    "3x3_1x4x24x24_gp4_nobias": lambda: dw_conv2d_3x3_1x4x24x24_gp4_nobias,
    "3x3_1x4x24x24_gp4_st1": lambda: dw_conv2d_3x3_1x4x24x24_gp4_st1,
    "3x3_2x8x27x27_gp8_st3": lambda: dw_conv2d_3x3_2x8x27x27_gp8_st3,
    "two_dw_conv2d": lambda: two_dw_conv2d,
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_conv2d_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_conv2d_FP.items()
    for q in [True, False]
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_conv2d_u85 = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in {
        "2x2_1x6x4x4_gp6_st1": lambda: dw_conv2d_2x2_1x6x4x4_gp6_st1,
        "3x3_1x3x24x24_gp3_st1": lambda: dw_conv2d_3x3_1x3x24x24_gp3_st1,
        "3x3_1x4x24x24_gp4_st1": lambda: dw_conv2d_3x3_1x4x24x24_gp4_st1,
        "3x3_1x4x24x24_gp4_nobias": lambda: dw_conv2d_3x3_1x4x24x24_gp4_nobias,
    }.items()
    for q in [True, False]
}

test_data_conv1d_FP = {
    "2_1x6x4_gp6_st1": lambda: dw_conv1d_2_1x6x4_gp6_st1,
    "two_dw_conv1d": lambda: two_dw_conv1d,
    "3_1x3x32_gp3_st1": lambda: dw_conv1d_3_1x3x32_gp3_st1,
    "3_1x3x14_gp3_st1": lambda: dw_conv1d_3_1x3x14_gp3_st1,
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_conv1d_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_conv1d_FP.items()
    for q in [True, False]
}


def _get_dtype_count(model: torch.nn.Module):
    nbr_convs: int = model.nbr_convs  # noqa
    return {
        "CONST": {"INT4": nbr_convs * 2},
        "DEPTHWISE_CONV2D": {"INT32": nbr_convs},
        "RESCALE": {"INT8": nbr_convs},
    }


@common.parametrize("test_data", test_data_conv1d_FP | test_data_conv2d_FP)
def test_convolution_2d_tosa_FP_depthwise(test_data: torch.nn.Module):
    pipeline = TosaPipelineFP[input_t](
        test_data(),
        test_data().get_inputs(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
@common.parametrize("test_data", test_data_conv1d_INT | test_data_conv2d_INT)
def test_convolution_2d_tosa_INT_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_conv1d_INT | test_data_conv2d_INT)
def test_convolution_2d_tosa_INT_a8w4_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=exir_op,
        tosa_extensions=["int4"],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check_dtype_count,
        _get_dtype_count(model),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_conv1d_FP | test_data_conv2d_FP)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_no_quant_depthwise(test_data: torch.nn.Module):
    model = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_conv1d_INT | test_data_conv2d_INT)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_quant_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=exir_op,
        quantize=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_conv2d_INT)
def test_convolution_2d_u55_INT_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_conv2d_INT)
def test_convolution_2d_u55_a8w4_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_conv1d_INT)
def test_convolution_1d_u55_INT_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_conv1d_INT)
def test_convolution_1d_u55_a8w4_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_conv2d_INT)
def test_convolution_2d_u85_INT_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_conv2d_INT)
def test_convolution_2d_u85_a8w4_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_conv1d_INT)
def test_convolution_1d_u85_INT_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_conv1d_INT)
def test_convolution_1d_u85_a8w4_depthwise(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()
