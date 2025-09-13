# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
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
        run_on_fvp=True,
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
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("per_channel_quantization", quant_test_data)
def test_mv2_vgf_INT(per_channel_quantization):
    pipeline = VgfPipeline[input_t](
        mv2,
        model_inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.25,
        qtol=1,
    )
    # TODO: MLETORCH-1167 Create Vulkan backend e2e tests
    # pipeline.change_args(
    #     "run_method_and_compare_outputs", get_test_inputs(), atol=3e-1, qtol=1.0
    # )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_mv2_vgf_FP():
    pipeline = VgfPipeline[input_t](
        mv2,
        model_inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=True,
    )
    # TODO: MLETORCH-1167 Create Vulkan backend e2e tests
    # pipeline.change_args(
    #     "run_method_and_compare_outputs", get_test_inputs(), atol=3e-1, qtol=1.0
    # )  # TODO: MLETORCH-1036 decrease tolerance
    pipeline.run()
