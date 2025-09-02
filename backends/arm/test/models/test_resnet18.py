# Copyright 2025 Arm Limited and/or its affiliates.
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
)

from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights)
model = model.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model_inputs = (normalize(torch.randn((1, 3, 224, 224))),)

input_t = Tuple[torch.Tensor]


quant_test_data = {
    "per_channel_quantization=true": True,
    "per_channel_quantization=false": False,
}


def test_resnet_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("per_channel_quantization", quant_test_data)
def test_resnet_tosa_INT(per_channel_quantization):
    pipeline = TosaPipelineINT[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.5,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone300
@common.parametrize("per_channel_quantization", quant_test_data)
def test_resnet_u55_INT(per_channel_quantization):
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.5,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@pytest.mark.xfail(
    reason="For resnet18 for Ethos-U85, the SRAM memory footprint is very high. The compiler team is investigating."
)
@common.XfailIfNoCorstone320
@common.parametrize("per_channel_quantization", quant_test_data)
def test_resnet_u85_INT(per_channel_quantization):
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        per_channel_quantization=per_channel_quantization,
        atol=0.5,
        qtol=1,
    )
    pipeline.run()
