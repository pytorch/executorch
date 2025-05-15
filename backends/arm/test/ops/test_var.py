# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
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


@common.parametrize("test_data", Var.test_parameters)
def test_var_dim_tosa_MI_no_dim(test_data: Tuple):
    test_data, keepdim, correction = test_data()
    pipeline = TosaPipelineMI[input_t1](
        Var(keepdim, correction),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Var.test_parameters)
def test_var_dim_tosa_BI_no_dim(test_data: Tuple):
    test_data, keepdim, correction = test_data()
    pipeline = TosaPipelineBI[input_t1](
        Var(keepdim, correction),
        (test_data,),
        aten_op=[],
        exir_op=[],
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", Var.test_parameters)
@common.XfailIfNoCorstone300
def test_var_dim_u55_BI_no_dim(test_data: Tuple):
    test_data, keepdim, correction = test_data()
    pipeline = EthosU55PipelineBI[input_t1](
        Var(keepdim, correction),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", Var.test_parameters)
@common.XfailIfNoCorstone320
def test_var_dim_u85_BI_no_dim(test_data: Tuple):
    test_data, keepdim, correction = test_data()
    pipeline = EthosU85PipelineBI[input_t1](
        Var(keepdim, correction),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters)
def test_var_dim_tosa_MI(test_data: Tuple):
    test_data, dim, keepdim, unbiased = test_data()
    pipeline = TosaPipelineMI[input_t1](
        VarDim(dim, keepdim, unbiased),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters)
def test_var_dim_tosa_BI(test_data: Tuple):

    test_data, dim, keepdim, unbiased = test_data()
    pipeline = TosaPipelineBI[input_t1](
        VarDim(dim, keepdim, unbiased),
        (test_data,),
        aten_op=[],
        exir_op=[],
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters_u55)
@common.XfailIfNoCorstone300
def test_var_dim_u55_BI(test_data: Tuple):
    test_data, dim, keepdim, unbiased = test_data()
    pipeline = EthosU55PipelineBI[input_t1](
        VarDim(dim, keepdim, unbiased),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", VarDim.test_parameters)
@common.XfailIfNoCorstone320
def test_var_dim_u85_BI(test_data: Tuple):
    test_data, dim, keepdim, unbiased = test_data()
    pipeline = EthosU85PipelineBI[input_t1](
        VarDim(dim, keepdim, unbiased),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", VarCorrection.test_parameters)
def test_var_dim_tosa_MI_correction(test_data: Tuple):
    test_data, dim, keepdim, correction = test_data()
    pipeline = TosaPipelineMI[input_t1](
        VarCorrection(dim, keepdim, correction),
        (test_data,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", VarCorrection.test_parameters)
def test_var_dim_tosa_BI_correction(test_data: Tuple):
    test_data, dim, keepdim, correction = test_data()
    pipeline = TosaPipelineBI[input_t1](
        VarCorrection(dim, keepdim, correction),
        (test_data,),
        aten_op=[],
        exir_op=[],
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", VarCorrection.test_parameters)
@common.XfailIfNoCorstone300
def test_var_dim_u55_BI_correction(test_data: Tuple):
    test_data, dim, keepdim, correction = test_data()
    pipeline = EthosU55PipelineBI[input_t1](
        VarCorrection(dim, keepdim, correction),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_data", VarCorrection.test_parameters)
@common.XfailIfNoCorstone320
def test_var_dim_u85_BI_correction(test_data: Tuple):
    test_data, dim, keepdim, correction = test_data()
    pipeline = EthosU85PipelineBI[input_t1](
        VarCorrection(dim, keepdim, correction),
        (test_data,),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()
