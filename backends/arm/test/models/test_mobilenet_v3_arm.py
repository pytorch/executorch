# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import common
import pytest

import torch

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

from torchvision import models, transforms

mv3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights)
mv3 = mv3.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

input_tensor = torch.rand(1, 3, 232, 232)

model_inputs = (normalize(input_tensor),)
input_t = Tuple[torch.Tensor]


@pytest.mark.slow
def test_mv3_tosa_MI():
    pipeline = TosaPipelineMI[input_t](
        mv3, model_inputs, aten_op=[], exir_op=[], use_to_edge_transform_and_lower=True
    )
    pipeline.run()


@pytest.mark.slow
def test_mv3_tosa_BI():
    pipeline = TosaPipelineBI[input_t](
        mv3,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=0.5,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone300
def test_mv3_u55_BI():
    pipeline = EthosU55PipelineBI[input_t](
        mv3,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        atol=0.5,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone320
def test_mv3_u85_BI():
    pipeline = EthosU85PipelineBI[input_t](
        mv3,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        atol=0.5,
        qtol=1,
    )
    pipeline.run()
