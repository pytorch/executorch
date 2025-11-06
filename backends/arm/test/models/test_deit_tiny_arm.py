# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import Tuple

import timm  # type: ignore[import-untyped]

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

from timm.data import (  # type: ignore[import-untyped]
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torchvision import transforms  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


deit_tiny = timm.models.deit.deit_tiny_patch16_224(pretrained=True)
deit_tiny.eval()

normalize = transforms.Normalize(
    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
)
model_inputs = (normalize(torch.rand((1, 3, 224, 224))),)

input_t = Tuple[torch.Tensor]


def test_deit_tiny_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


def test_deit_tiny_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=1.5,
        qtol=1,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_deit_tiny_vgf_INT():
    pipeline = VgfPipeline[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        use_to_edge_transform_and_lower=True,
        atol=1.5,
        qtol=1,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_deit_tiny_vgf_FP():
    pipeline = VgfPipeline[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
