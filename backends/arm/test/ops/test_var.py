# Copyright 2024-2025 Arm Limited and/or its affiliates.
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

input_t1 = Tuple[torch.Tensor]  # Input x


class Var(torch.nn.Module):
    test_parameters = {
        "var_4d_keep_dim_0_correction": lambda: (torch.randn(1, 50, 10, 20), True, 0),
        "var_3d_no_keep_dim_0_correction": lambda: (torch.rand(1, 50, 10), False, 0),
        "var_4d_keep_dim_1_correction": lambda: (torch.randn(1, 30, 15, 20), True, 1),
        "var_4d_no_keep_dim_0_5_correction": lambda: (
            torch.rand(1, 50, 10, 20),
            False,
            0.5,
        ),
    }

    def __init__(self, keepdim: bool = True, correction: int = 0):
        super().__init__()
        self.keepdim = keepdim
        self.correction = correction

    def forward(
        self,
        x: torch.Tensor,
    ):
        return x.var(keepdim=self.keepdim, correction=self.correction)


class VarDim(torch.nn.Module):
    test_parameters = {
        "var_4d_dim_1_keep_dim_unbiased": lambda: (
            torch.randn(1, 50, 10, 20),
            1,
            True,
            False,
        ),
        "var_3d_dim_neg_2_no_keep_dim_unbiased": lambda: (
            torch.rand(1, 50, 10),
            -2,
            False,
            False,
        ),
        "var_3d_dim_neg_3_keep_dim_biased": lambda: (
            torch.randn(1, 30, 15, 20),
            -3,
            True,
            True,
        ),
        "var_3d_dim_neg_1_no_keep_dim_biased": lambda: (
            torch.rand(1, 50, 10, 20),
            -1,
            False,
            True,
        ),
    }

    test_parameters_u55 = {
        "var_4d_dim_1_keep_dim_unbiased": lambda: (
            torch.randn(1, 50, 10, 20),
            1,
            True,
            False,
        ),
        "var_4d_dim_neg_3_keep_dim_biased": lambda: (
            torch.randn(1, 30, 15, 20),
            -3,
            True,
            True,
        ),
    }

    test_parameters_u55_xfails = {
        "var_3d_dim_neg_2_keep_dim_unbiased": lambda: (
            torch.rand(1, 50, 10),
            -2,
            True,
            False,
        ),
        "var_3d_dim_neg_1_keep_dim_biased": lambda: (
            torch.rand(1, 50, 10, 20),
            -1,
            True,
            True,
        ),
    }

    def __init__(self, dim: int = -1, keepdim: bool = True, unbiased: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.unbiased = unbiased

    def forward(
        self,
        x: torch.Tensor,
    ):
        return x.var(dim=self.dim, keepdim=self.keepdim, unbiased=self.unbiased)


class VarCorrection(torch.nn.Module):
    test_parameters = {
        "var_4d_dims_keep_dim_0_correction": lambda: (
            torch.randn(1, 50, 10, 20),
            (-1, -2),
            True,
            0,
        ),
        "var_3d_dims_keep_dim_0_correction": lambda: (
            torch.rand(1, 50, 10),
            (-2),
            True,
            0,
        ),
        "var_4d_dims_keep_dim_1_correction": lambda: (
            torch.randn(1, 30, 15, 20),
            (-1, -2, -3),
            True,
            1,
        ),
        "var_4d_dims_keep_dim_0_5_correction": lambda: (
            torch.rand(1, 50, 10, 20),
            (-1, -2),
            True,
            0.5,
        ),
    }

    def __init__(self, dim: int = -1, keepdim: bool = True, correction: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.correction = correction

    def forward(
        self,
        x: torch.Tensor,
    ):
        return x.var(dim=self.dim, keepdim=self.keepdim, correction=self.correction)


##########
## Var ###
##########


@common.parametrize("test_data", Var.test_parameters)
def test_var_dim_tosa_FP_no_dim(test_data: Tuple):
    test_data, keepdim, correction = test_data()
    pipeline = TosaPipelineFP[input_t1](
        Var(keepdim, correction),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Var.test_parameters)
def test_var_dim_tosa_INT_no_dim(test_data: Tuple):
    test_data, keepdim, correction = test_data()
    pipeline = TosaPipelineINT[input_t1](
        Var(keepdim, correction),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Var.test_parameters)
@common.XfailIfNoCorstone300
def test_var_dim_u55_INT_no_dim(test_data: Tuple):
    test_data, keepdim, correction = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        Var(keepdim, correction),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", Var.test_parameters)
@common.XfailIfNoCorstone320
def test_var_dim_u85_INT_no_dim(test_data: Tuple):
    test_data, keepdim, correction = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        Var(keepdim, correction),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", Var.test_parameters)
@common.SkipIfNoModelConverter
def test_var_dim_no_dim_vgf_no_quant(test_data: Tuple):
    data, keepdim, correction = test_data()
    pipeline = VgfPipeline[input_t1](
        Var(keepdim, correction),
        (data,),
        [],
        [],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", Var.test_parameters)
@common.SkipIfNoModelConverter
def test_var_dim_no_dim_vgf_quant(test_data: Tuple):
    data, keepdim, correction = test_data()
    pipeline = VgfPipeline[input_t1](
        Var(keepdim, correction),
        (data,),
        [],
        [],
        quantize=True,
    )
    pipeline.run()


#############
## VarDim ###
#############


@common.parametrize("test_data", VarDim.test_parameters)
def test_var_dim_tosa_FP(test_data: Tuple):
    test_data, dim, keepdim, unbiased = test_data()
    pipeline = TosaPipelineFP[input_t1](
        VarDim(dim, keepdim, unbiased),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters)
def test_var_dim_tosa_INT(test_data: Tuple):

    test_data, dim, keepdim, unbiased = test_data()
    pipeline = TosaPipelineINT[input_t1](
        VarDim(dim, keepdim, unbiased),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters_u55)
@common.XfailIfNoCorstone300
def test_var_dim_u55_INT(test_data: Tuple):
    test_data, dim, keepdim, unbiased = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        VarDim(dim, keepdim, unbiased),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters)
@common.XfailIfNoCorstone320
def test_var_dim_u85_INT(test_data: Tuple):
    test_data, dim, keepdim, unbiased = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        VarDim(dim, keepdim, unbiased),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters)
@common.SkipIfNoModelConverter
def test_var_dim_vgf_no_quant(test_data: Tuple):
    data, dim, keepdim, unbiased = test_data()
    pipeline = VgfPipeline[input_t1](
        VarDim(dim, keepdim, unbiased),
        (data,),
        [],
        [],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters)
@common.SkipIfNoModelConverter
def test_var_dim_vgf_quant(test_data: Tuple):
    data, dim, keepdim, unbiased = test_data()
    pipeline = VgfPipeline[input_t1](
        VarDim(dim, keepdim, unbiased),
        (data,),
        [],
        [],
        quantize=True,
    )
    pipeline.run()


####################
## VarCorrection ###
####################


@common.parametrize("test_data", VarCorrection.test_parameters)
def test_var_dim_tosa_FP_correction(test_data: Tuple):
    test_data, dim, keepdim, correction = test_data()
    pipeline = TosaPipelineFP[input_t1](
        VarCorrection(dim, keepdim, correction),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarCorrection.test_parameters)
def test_var_dim_tosa_INT_correction(test_data: Tuple):
    test_data, dim, keepdim, correction = test_data()
    pipeline = TosaPipelineINT[input_t1](
        VarCorrection(dim, keepdim, correction),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


# TODO: Xfail "var_3d_dims_keep_dim_0_correction" until the Ethos-U Vela compiler ships commit
# 642f7517d3a6bd053032e1942822f6e38ccd546f. That patch fixes the bug that causes the test to fail.
@common.parametrize(
    "test_data",
    VarCorrection.test_parameters,
    xfails={
        "var_3d_dims_keep_dim_0_correction": (
            "Blocked by Vela commit 642f7517d3a6bd053032e1942822f6e38ccd546f"
        ),
    },
)
@common.XfailIfNoCorstone300
def test_var_dim_u55_INT_correction(test_data: Tuple):
    test_data, dim, keepdim, correction = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        VarCorrection(dim, keepdim, correction),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarCorrection.test_parameters)
@common.XfailIfNoCorstone320
def test_var_dim_u85_INT_correction(test_data: Tuple):
    test_data, dim, keepdim, correction = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        VarCorrection(dim, keepdim, correction),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarCorrection.test_parameters)
@common.SkipIfNoModelConverter
def test_var_dim_correction_vgf_no_quant(test_data: Tuple):
    data, dim, keepdim, corr = test_data()
    pipeline = VgfPipeline[input_t1](
        VarCorrection(dim, keepdim, corr),
        (data,),
        [],
        [],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", VarCorrection.test_parameters)
@common.SkipIfNoModelConverter
def test_var_dim_correction_vgf_quant(test_data: Tuple):
    data, dim, keepdim, corr = test_data()
    pipeline = VgfPipeline[input_t1](
        VarCorrection(dim, keepdim, corr),
        (data,),
        [],
        [],
        quantize=True,
    )
    pipeline.run()
