# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.quantizer import get_symmetric_quantization_config
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from torchvision import models, transforms  # type: ignore[import-untyped]
from torchvision.models.mobilenetv2 import (  # type: ignore[import-untyped]
    MobileNet_V2_Weights,
)


mv2 = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
mv2 = mv2.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model_inputs = (normalize(torch.rand((1, 3, 224, 224))),)
input_t = Tuple[torch.Tensor]


quant_test_data = {
    "per_channel_quantization=true": True,
    "per_channel_quantization=false": False,
}


def _use_partial_quantizer(pipeline):
    """Set the pipeline's quantizer to only include Conv2d and ReLU6"""
    quant_cfg = get_symmetric_quantization_config()
    pipeline.quantizer.set_global(None)
    pipeline.quantizer.set_module_type(torch.nn.Conv2d, quant_cfg)
    pipeline.quantizer.set_module_type(torch.nn.ReLU6, quant_cfg)


def test_mv2_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        mv2, model_inputs, aten_op=[], exir_op=[], use_to_edge_transform_and_lower=True
    )
    pipeline.run()


def test_mv2_tosa_FP_channels_last():
    input_tensor = model_inputs[0].to(memory_format=torch.channels_last)
    pipeline = TosaPipelineFP[input_t](
        mv2,
        (input_tensor,),
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    # Changing memory format leads to an unsupported as_strided_copy op being inserted into the graph,
    # leading to a graph break.
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 2}
    )
    pipeline.run()


@common.parametrize("per_channel_quantization", quant_test_data)
def test_mv2_tosa_INT(per_channel_quantization):
    pipeline = TosaPipelineINT[input_t](
        mv2,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone300
@common.parametrize("per_channel_quantization", quant_test_data)
def test_mv2_u55_INT(per_channel_quantization):
    pipeline = EthosU55PipelineINT[input_t](
        mv2,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone320
@common.parametrize("per_channel_quantization", quant_test_data)
def test_mv2_u85_INT(per_channel_quantization):
    pipeline = EthosU85PipelineINT[input_t](
        mv2,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("per_channel_quantization", quant_test_data)
def test_mv2_vgf_quant(per_channel_quantization):
    pipeline = VgfPipeline[input_t](
        mv2,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_mv2_vgf_no_quant():
    pipeline = VgfPipeline[input_t](
        mv2,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        quantize=False,
    )
    pipeline.run()


def test_mv2_partial_quant_tosa_INT_FP():
    pipeline = TosaPipelineINT[input_t](
        mv2,
        model_inputs,
        aten_op=[],
        exir_op=[],
        tosa_extensions=["FP"],
        use_to_edge_transform_and_lower=True,
        atol=0.20,
    )
    _use_partial_quantizer(pipeline)
    pipeline.run()


@common.SkipIfNoModelConverter
def test_mv2_partial_quant_vgf_quant():
    pipeline = VgfPipeline[input_t](
        mv2,
        model_inputs,
        aten_op=[],
        exir_op=[],
        quantize=True,
        atol=0.10,
    )
    _use_partial_quantizer(pipeline)
    pipeline.run()
