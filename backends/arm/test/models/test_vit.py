# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

from torchvision import models, transforms

vit_b_16_model = models.vit_b_16(weights="IMAGENET1K_V1")
vit = vit_b_16_model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

input_tensor = torch.rand(1, 3, 224, 224)

model_inputs = (normalize(input_tensor),)
input_t = Tuple[torch.Tensor]


@pytest.mark.slow
def test_vit_tosa_MI():
    pipeline = TosaPipelineMI[input_t](
        vit, model_inputs, aten_op=[], exir_op=[], use_to_edge_transform_and_lower=True
    )
    pipeline.run()


@pytest.mark.slow
def test_vit_tosa_BI():
    pipeline = TosaPipelineBI[input_t](
        vit,
        model_inputs,
        aten_op=[],
        exir_op=[],
        atol=5.0,
        qtol=1,
    )

    pipeline.run()


@pytest.mark.slow
@pytest.mark.xfail(reason="Unsupported transpose")
def test_vit_u55_BI():
    pipeline = EthosU55PipelineBI[input_t](
        vit,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@pytest.mark.slow
def test_vit_u85_BI():
    pipeline = EthosU85PipelineBI[input_t](
        vit,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()
