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
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
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


def test_mv2_tosa_MI():
    pipeline = TosaPipelineMI[input_t](
        mv2, model_inputs, aten_op=[], exir_op=[], use_to_edge_transform_and_lower=True
    )
    pipeline.run()


def test_mv2_tosa_BI():
    pipeline = TosaPipelineBI[input_t](
        mv2,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone300
def test_mv2_u55_BI():
    pipeline = EthosU55PipelineBI[input_t](
        mv2,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()


@pytest.mark.slow
@common.XfailIfNoCorstone320
def test_mv2_u85_BI():
    pipeline = EthosU85PipelineBI[input_t](
        mv2,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
        atol=0.25,
        qtol=1,
    )
    pipeline.run()
