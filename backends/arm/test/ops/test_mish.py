# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for Mish op."""

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    VgfPipeline,
)


class _MishOpSupportModule(torch.nn.Module):
    """Minimal Mish example for operator-support documentation coverage."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mish(x)


@common.XfailIfNoCorstone320
def test_mish_u85_INT():
    """Quantized Mish must lower as a single Ethos-U85 delegate (no FVP)."""
    torch.manual_seed(0)
    input_tensor = torch.randn(1, 16, 32)
    pipeline = EthosU85PipelineINT(
        _MishOpSupportModule().eval(),
        (input_tensor,),
        aten_ops="torch.ops.aten.mish.default",
        exir_ops=[],
    )
    pipeline.run()


def test_mish_u55_INT():
    x = torch.randn(1, 16, 32)

    pipeline = EthosU55PipelineINT(
        _MishOpSupportModule().eval(),
        (x,),
        aten_ops="torch.ops.aten.mish.default",
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_mish_vgf_quant():
    x = torch.randn(1, 16, 32)

    pipeline = VgfPipeline(
        _MishOpSupportModule().eval(),
        (x,),
        aten_op="torch.ops.aten.mish.default",
        exir_op=[],
        quantize=True,
    )
    pipeline.run()


def test_mish_tosa_FP():
    """Validate floating-point Mish lowering and TOSA reference execution."""
    inputs = (torch.linspace(-6.0, 6.0, 32).reshape(1, 1, 32),)
    pipeline = TosaPipelineFP(
        _MishOpSupportModule(),
        inputs,
        aten_op="torch.ops.aten.mish.default",
        exir_op=[],
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_mish_vgf_no_quant():
    """Validate floating-point Mish execution through VGF."""
    inputs = (torch.linspace(-6.0, 6.0, 32).reshape(1, 1, 32),)
    pipeline = VgfPipeline(
        _MishOpSupportModule(),
        inputs,
        aten_op="torch.ops.aten.mish.default",
        exir_op=[],
        quantize=False,
    )
    pipeline.run()
