# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Residual conv block model test for ARM TOSA backend.

Tests a minimal residual architecture with conv->batchnorm->relu->add blocks and
permute operations, representative of quantized signal processing models where
FuseConsecutiveRescalesPass eliminates redundant RESCALE pairs.

"""

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


class ResidualConvBlock(torch.nn.Module):
    """Residual conv block with batchnorm and permute operations.

    Architecture: conv->bn->relu->add (residual) -> permute ->
    conv->bn->relu->add. When quantized, each residual add is
    wrapped with INT32 RESCALEs by InsertRescaleInt32Pass. Stacked
    blocks create consecutive RESCALE pairs (INT32->INT8->INT32)
    between adjacent adds that FuseConsecutiveRescalesPass
    eliminates.

    """

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        # Block 1: conv → batchnorm → relu → residual add
        out = self.relu1(self.bn1(self.conv1(x)))
        out = out + x  # residual add 1

        # Channel reordering (common in signal processing models)
        out = out.permute(0, 1, 3, 2)

        # Block 2: conv → batchnorm → relu → residual add
        out2 = self.relu2(self.bn2(self.conv2(out)))
        out2 = out2 + out  # residual add 2
        return out2


model = ResidualConvBlock().eval()
model_inputs = (torch.randn(1, 3, 8, 8),)
input_t = Tuple[torch.Tensor]


def test_residual_conv_block_tosa_FP():
    pipeline = TosaPipelineFP[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


def test_residual_conv_block_tosa_INT():
    pipeline = TosaPipelineINT[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        atol=0.25,
        qtol=1,
        frobenius_threshold=None,
        cosine_threshold=None,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_residual_conv_block_u55_INT():
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_residual_conv_block_u85_INT():
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model_inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_residual_conv_block_vgf_quant():
    pipeline = VgfPipeline[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_residual_conv_block_vgf_no_quant():
    pipeline = VgfPipeline[input_t](
        model,
        model_inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        quantize=False,
    )
    pipeline.run()
