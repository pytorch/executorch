# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

rsub_test_data = {
    "rand_2D_4x4": lambda: (torch.rand(4, 4), 2),
    "rand_3D_4x4x4": lambda: (torch.rand(4, 2, 2), 1.5),
    "rand_4D_2x2x4x4": lambda: (torch.rand(2, 2, 4, 4), -1.1),
    "rand_4D_big_small": lambda: (
        (10e30) * torch.randn(1, 20, 30, 40),
        -0.25,
    ),
    "zero": lambda: (torch.rand(4, 4), 0),
    # "swapped": lambda: (2, torch.rand(4, 4)), # torch.rsub(Scalar, Tensor) is not supported as it is not supported in eager mode.
}


class Rsub(torch.nn.Module):
    aten_op = "torch.ops.aten.rsub.Scalar"
    exir_op = "executorch_exir_dialects_edge__ops_aten_sub_Tensor"

    def forward(self, x: torch.Tensor, y: int):
        return torch.rsub(x, y)


class RsubAlpha(torch.nn.Module):
    """Driven through the pipeline rather than the rewrite alone: the alpha the
    rewrite produces only disappears once the decomposition after it runs.
    """

    aten_op = "torch.ops.aten.rsub.Scalar"
    exir_op = "executorch_exir_dialects_edge__ops_aten_sub_Tensor"

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: int):
        return torch.rsub(x, y, alpha=self.alpha)


rsub_alpha_test_data = {
    "alpha_int": (lambda: (torch.rand(4, 4), 2), 2),
    "alpha_negative": (lambda: (torch.rand(4, 4), 1.5), -2),
    "alpha_fractional": (lambda: (torch.rand(2, 2, 4, 4), -1.1), 0.5),
    "alpha_one_float": (lambda: (torch.rand(4, 4), 2), 1.0),
    "alpha_one_int": (lambda: (torch.rand(4, 4), 2), 1),
}


input_t1 = Tuple[torch.Tensor, torch.Tensor]


@common.parametrize("test_data", rsub_test_data)
def test_rsub_scalar_tosa_FP(test_data):
    pipeline = TosaPipelineFP[input_t1](
        Rsub(),
        test_data(),
        aten_op=Rsub.aten_op,
        exir_op=Rsub.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", rsub_test_data)
def test_rsub_scalar_tosa_INT(test_data):
    """Test Subtraction (TOSA INT)"""
    pipeline = TosaPipelineINT[input_t1](
        Rsub(),
        test_data(),
        aten_op="torch.ops.aten.sub.Tensor",
        exir_op=Rsub.exir_op,
        qtol=0,
        cosine_threshold=None,  # For rand_4D_big_small, the output diff is large which throws off the cosine similarity even if it is relatively small.
    )
    pipeline.run()


@common.parametrize("test_data", rsub_alpha_test_data)
def test_rsub_scalar_alpha_tosa_FP(test_data):
    data, alpha = test_data
    pipeline = TosaPipelineFP[input_t1](
        RsubAlpha(alpha),
        data(),
        aten_op=RsubAlpha.aten_op,
        exir_op=RsubAlpha.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", rsub_alpha_test_data)
def test_rsub_scalar_alpha_tosa_INT(test_data):
    data, alpha = test_data
    pipeline = TosaPipelineINT[input_t1](
        RsubAlpha(alpha),
        data(),
        aten_op="torch.ops.aten.sub.Tensor",
        exir_op=RsubAlpha.exir_op,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", rsub_test_data)
@common.XfailIfNoCorstone300
def test_rsub_scalar_u55_INT(test_data):
    """Test Subtraction on Ethos-U55 (FVP Mode)"""
    pipeline = EthosU55PipelineINT[input_t1](
        Rsub(),
        test_data(),
        aten_ops="torch.ops.aten.sub.Tensor",
        exir_ops=Rsub.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", rsub_test_data)
@common.XfailIfNoCorstone320
def test_rsub_scalar_u85_INT(test_data):
    """Test Subtraction on Ethos-U85 (FVP Mode)"""
    pipeline = EthosU85PipelineINT[input_t1](
        Rsub(),
        test_data(),
        aten_ops="torch.ops.aten.sub.Tensor",
        exir_ops=Rsub.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", rsub_alpha_test_data)
@common.XfailIfNoCorstone300
def test_rsub_scalar_alpha_u55_INT(test_data):
    """Test Subtraction with alpha on Ethos-U55 (FVP Mode)"""
    data, alpha = test_data
    pipeline = EthosU55PipelineINT[input_t1](
        RsubAlpha(alpha),
        data(),
        aten_ops="torch.ops.aten.sub.Tensor",
        exir_ops=RsubAlpha.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", rsub_alpha_test_data)
@common.XfailIfNoCorstone320
def test_rsub_scalar_alpha_u85_INT(test_data):
    """Test Subtraction with alpha on Ethos-U85 (FVP Mode)"""
    data, alpha = test_data
    pipeline = EthosU85PipelineINT[input_t1](
        RsubAlpha(alpha),
        data(),
        aten_ops="torch.ops.aten.sub.Tensor",
        exir_ops=RsubAlpha.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", rsub_test_data)
@common.SkipIfNoModelConverter
def test_rsub_scalar_vgf_no_quant(test_data: Tuple[torch.Tensor]):
    """Test Subtraction (VGF FP)"""
    pipeline = VgfPipeline[input_t1](
        Rsub(),
        test_data(),
        Rsub.aten_op,
        Rsub.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", rsub_test_data)
@common.SkipIfNoModelConverter
def test_rsub_scalar_vgf_quant(test_data: Tuple[torch.Tensor]):
    """Test Subtraction (VGF INT)"""
    pipeline = VgfPipeline[input_t1](
        Rsub(),
        test_data(),
        aten_op="torch.ops.aten.sub.Tensor",
        exir_op=Rsub.exir_op,
        quantize=True,
    )
    pipeline.run()
