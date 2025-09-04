# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple, Union

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.div.Tensor"
exir_op = "executorch_exir_dialects_edge__ops_aten_div_Tensor"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, input, other, rounding_mode) See torch.div() for info
    "op_div_rank1_ones": lambda: (torch.ones(5), torch.ones(5), None),
    "op_div_rank1_negative_ones": lambda: (
        torch.ones(5) * (-1),
        torch.ones(5) * (-1),
        None,
    ),
    "op_div_rank1_rand": lambda: (
        torch.rand(5) * 5,
        torch.rand(5) * 5,
        None,
    ),
    "op_div_rank4_ones": lambda: (
        torch.ones(5, 10, 25, 20),
        torch.ones(5, 10, 25, 20),
        None,
    ),
    "op_div_rank4_negative_ones": lambda: (
        (-1) * torch.ones(5, 10, 25, 20),
        torch.ones(5, 10, 25, 20),
        None,
    ),
    "op_div_rank4_ones_div_negative": lambda: (
        torch.ones(5, 10, 25, 20),
        (-1) * torch.ones(5, 10, 25, 20),
        None,
    ),
    "op_div_rank4_large_rand": lambda: (
        200 * torch.rand(5, 10, 25, 20),
        torch.rand(5, 10, 25, 20),
        None,
    ),
    "op_div_rank4_negative_large_rand": lambda: (
        (-200) * torch.rand(5, 10, 25, 20),
        torch.rand(5, 10, 25, 20),
        None,
    ),
    "op_div_rank4_large_randn": lambda: (
        200 * torch.randn(5, 10, 25, 20) + 1,
        torch.rand(5, 10, 25, 20) + 1,
        None,
    ),
    "op_div_rank4_randn_mutltiple_broadcasts": lambda: (
        torch.randn(1, 4, 4, 1),
        torch.randn(1, 1, 4, 4),
        None,
    ),
}


class Div(torch.nn.Module):

    def forward(
        self,
        input_: Union[torch.Tensor, torch.types.Number],
        other_: Union[torch.Tensor, torch.types.Number],
        rounding_mode: Optional[str] = None,
    ):
        if rounding_mode is None:
            return torch.div(input=input_, other=other_)
        else:
            return torch.div(input=input_, other=other_, rounding_mode=rounding_mode)


@common.parametrize("test_data", test_data_suite)
def test_div_tensor_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](Div(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_div_tensor_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](Div(), test_data(), aten_op=[], exir_op=[])
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_div_tensor_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Div(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_div_tensor_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Div(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_div_tensor_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Div(), test_data(), aten_op, exir_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_div_tensor_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Div(),
        test_data(),
        aten_op=[],
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
