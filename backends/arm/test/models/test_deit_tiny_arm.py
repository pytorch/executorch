# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import Tuple

import timm

import torch

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineBI,
    TosaPipelineMI,
)

from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


deit_tiny = timm.models.deit.deit_tiny_patch16_224(pretrained=True)
deit_tiny.eval()

normalize = transforms.Normalize(
    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
)
model_inputs = (normalize(torch.rand((1, 3, 224, 224))),)

input_t = Tuple[torch.Tensor]


def test_deit_tiny_tosa_MI():
    pipeline = TosaPipelineMI[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


def test_deit_tiny_tosa_BI():
    pipeline = TosaPipelineBI[input_t](
        deit_tiny,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=2.5,  # This needs to go down: MLETORCH-956
        qtol=1,
    )
    pipeline.run()
