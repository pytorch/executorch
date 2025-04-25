# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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

aten_op = "torch.ops.aten.sub.Tensor"
exir_op = "executorch_exir_dialects_edge__ops_aten_sub_Tensor"

# Single-input subtraction (x - x)
sub_test_data = {
    "ones_1D_5": (torch.ones(5),),
    "ones_1D_50": (torch.ones(50),),
    "rand_1D_10": (torch.rand(10),),
    "rand_2D_5x5": (torch.rand(5, 5),),
    "rand_3D_5x5x5": (torch.rand(5, 5, 5),),
    "rand_4D_2x3x4x5": (torch.rand(2, 3, 4, 5),),
    "zeros": (torch.zeros(10),),
}

fvp_sub_xfails = {"rand_4D_2x3x4x5": "MLETORCH-517 : Multiple batches not supported"}

# Two-input subtraction (x - y)
sub2_test_data = {
    "rand_2D_4x4": (torch.rand(4, 4), torch.rand(4, 4)),
    "rand_3D_4x4x4": (torch.rand(4, 2, 2), torch.rand(4, 2, 2)),
    "rand_4D_2x2x4x4": (torch.rand(2, 2, 4, 4), torch.rand(2, 2, 4, 4)),
    "zeros": (torch.rand(4, 4), torch.zeros(4, 4)),
}
fvp_sub2_xfails = {"rand_4D_2x2x4x4": "MLETORCH-517 : Multiple batches not supported"}


class Sub(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x - x


class Sub2(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x - y


input_t1 = Tuple[torch.Tensor]  # Input x
input_t2 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


@common.parametrize("test_data", sub_test_data)
def test_sub_tosa_MI(test_data):
    """Test Subtraction (TOSA MI)"""
    pipeline = TosaPipelineMI[input_t1](
        Sub(),
        test_data,
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", sub2_test_data)
def test_sub_2_tosa_MI(test_data: Tuple[torch.Tensor, torch.Tensor]):
    """Test Two-Operand Subtraction (TOSA MI)"""
    pipeline = TosaPipelineMI[input_t2](
        Sub2(),
        test_data,
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", sub_test_data)
def test_sub_tosa_BI(test_data):
    """Test Subtraction (TOSA BI)"""
    pipeline = TosaPipelineBI[input_t1](
        Sub(),
        test_data,
        aten_op,
        exir_op,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", sub2_test_data)
def test_sub_2_tosa_BI(test_data: Tuple[torch.Tensor, torch.Tensor]):
    """Test Two-Operand Subtraction (TOSA BI)"""
    pipeline = TosaPipelineBI[input_t2](
        Sub2(),
        test_data,
        aten_op,
        exir_op,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", sub_test_data)
def test_sub_u55_BI(test_data):
    """Test Subtraction on Ethos-U55"""
    pipeline = EthosU55PipelineBI[input_t1](
        Sub(),
        test_data,
        aten_op,
        exir_op,
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", sub2_test_data)
def test_sub_2_u55_BI(test_data: Tuple[torch.Tensor, torch.Tensor]):
    """Test Two-Operand Subtraction on Ethos-U55"""
    pipeline = EthosU55PipelineBI[input_t2](
        Sub2(),
        test_data,
        aten_op,
        exir_op,
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", sub_test_data)
def test_sub_u85_BI(test_data):
    """Test Subtraction on Ethos-U85 (Quantized Mode)"""
    pipeline = EthosU85PipelineBI[input_t1](
        Sub(),
        test_data,
        aten_op,
        exir_op,
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", sub2_test_data)
def test_sub_2_u85_BI(test_data: Tuple[torch.Tensor, torch.Tensor]):
    """Test Two-Operand Subtraction on Ethos-U85"""
    pipeline = EthosU85PipelineBI[input_t2](
        Sub2(),
        test_data,
        aten_op,
        exir_op,
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", sub_test_data, fvp_sub_xfails)
@common.SkipIfNoCorstone300
def test_sub_u55_BI_on_fvp(test_data):
    """Test Subtraction on Ethos-U55 (FVP Mode)"""
    pipeline = EthosU55PipelineBI[input_t1](
        Sub(),
        test_data,
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", sub2_test_data, fvp_sub2_xfails)
@common.SkipIfNoCorstone300
def test_sub_2_u55_BI_on_fvp(test_data: Tuple[torch.Tensor, torch.Tensor]):
    """Test Two-Operand Subtraction on Ethos-U55 (FVP Mode)"""
    pipeline = EthosU55PipelineBI[input_t2](
        Sub2(),
        test_data,
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", sub_test_data, fvp_sub_xfails)
@common.SkipIfNoCorstone320
def test_sub_u85_BI_on_fvp(test_data):
    """Test Subtraction on Ethos-U85 (FVP Mode)"""
    pipeline = EthosU85PipelineBI[input_t1](
        Sub(),
        test_data,
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", sub2_test_data, fvp_sub2_xfails)
@common.SkipIfNoCorstone320
def test_sub_2_u85_BI_on_fvp(test_data: Tuple[torch.Tensor, torch.Tensor]):
    """Test Two-Operand Subtraction on Ethos-U85 (FVP Mode)"""
    pipeline = EthosU85PipelineBI[input_t2](
        Sub2(),
        test_data,
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()
