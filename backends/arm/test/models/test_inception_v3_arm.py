# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import common
import pytest

import torch

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from torchvision import models, transforms

ic3 = models.inception_v3(weights=models.Inception_V3_Weights)
ic3 = ic3.eval()

# Normalization values referenced from here:
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.quantization.inception_v3.html
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model_inputs = (normalize(torch.rand(1, 3, 224, 224)),)
input_t = Tuple[torch.Tensor]


@pytest.mark.slow
def test_ic3_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        ic3,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@pytest.mark.slow
def test_ic3_tosa_BI():
    pipeline = TosaPipelineINT[input_t](
        ic3,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=0.6,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@pytest.mark.skip(reason="Takes too long to run on CI")
@common.XfailIfNoCorstone300
def test_ic3_u55_BI():
    pipeline = EthosU55PipelineINT[input_t](
        ic3,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        atol=0.6,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@pytest.mark.skip(reason="Takes too long to run on CI")
@common.XfailIfNoCorstone320
def test_ic3_u85_BI():
    pipeline = EthosU85PipelineINT[input_t](
        ic3,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        atol=0.6,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@pytest.mark.skip(reason="Takes too long to run on CI")
@common.SkipIfNoModelConverter
def test_ic3_vgf_FP():
    pipeline = VgfPipeline[input_t](
        ic3,
        model_inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@pytest.mark.slow
@pytest.mark.skip(reason="Takes too long to run on CI")
@common.SkipIfNoModelConverter
def test_ic3_vgf_INT():
    pipeline = VgfPipeline[input_t](
        ic3,
        model_inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
