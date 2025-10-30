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

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(1, 10, 10, 10),
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "randn_pos": lambda: torch.randn(1, 4, 4, 4) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}

aten_op = "torch.ops.aten.exp.default"
input_t1 = Tuple[torch.Tensor]  # Input x


class Exp(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


@common.parametrize("test_data", test_data_suite)
def test_exp_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Exp(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_exp_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Exp(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_exp_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Exp(),
        (test_data(),),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_exp_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Exp(),
        (test_data(),),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_exp_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Exp(),
        (test_data(),),
        aten_op,
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_exp_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Exp(),
        (test_data(),),
        aten_op,
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
