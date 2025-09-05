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
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor]  # Input x

aten_op = "torch.ops.aten.cat.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_cat_default"


class Cat(torch.nn.Module):
    test_parameters = {
        "cat_ones_two_tensors": lambda: ((torch.ones(1), torch.ones(1)), 0),
        "cat_ones_and_rand_three_tensors": lambda: (
            (torch.ones(1, 2), torch.randn(1, 5), torch.randn(1, 1)),
            1,
        ),
        "cat_ones_and_rand_four_tensors": lambda: (
            (
                torch.ones(1, 2, 5),
                torch.randn(1, 2, 4),
                torch.randn(1, 2, 2),
                torch.randn(1, 2, 1),
            ),
            -1,
        ),
        "cat_rand_two_tensors": lambda: (
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 1)),
            3,
        ),
        "cat_rand_two_tensors_dim_0": lambda: (
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
            0,
        ),
        "cat_rand_two_tensors_dim_3": lambda: (
            (torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 1)),
            3,
        ),
        "cat_rand_large": lambda: (
            (
                10000 * torch.randn(2, 3, 1, 4),
                torch.randn(2, 7, 1, 4),
                torch.randn(2, 1, 1, 4),
            ),
            -3,
        ),
    }

    def __init__(self):
        super().__init__()

    def forward(self, t: tuple[torch.Tensor, ...], dim: int) -> torch.Tensor:
        return torch.cat(t, dim=dim)


@common.parametrize("test_data", Cat.test_parameters)
def test_cat_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


def test_cat_tosa_FP_4d():
    square = torch.ones((2, 2, 2, 2))
    for dim in range(-3, 3):
        test_data = ((square, square.clone()), dim)
        pipeline = TosaPipelineFP[input_t1](
            Cat(),
            test_data,
            aten_op,
            exir_op,
        )
        pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
def test_cat_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.XfailIfNoCorstone300
def test_cat_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.XfailIfNoCorstone320
def test_cat_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.SkipIfNoModelConverter
def test_cat_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cat(), test_data(), aten_op, exir_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", Cat.test_parameters)
@common.SkipIfNoModelConverter
def test_cat_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Cat(),
        test_data(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
