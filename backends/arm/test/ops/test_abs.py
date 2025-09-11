# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
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

aten_op = "torch.ops.aten.abs.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_abs_default"

input_t1 = Tuple[torch.Tensor]  # Input x


class Abs(torch.nn.Module):
    test_parameters = {
        "zeros": lambda: (torch.zeros(5),),
        "full": lambda: (torch.full((5,), -1, dtype=torch.float32),),
        "ones": lambda: (torch.ones(5) * -1,),
        "randn_1d": lambda: (torch.randn(8),),
        "randn_3d": lambda: (torch.randn(2, 3, 4),),
        "randn_4d": lambda: (torch.randn(1, 2, 3, 4),),
        "torch_normal": lambda: (torch.normal(mean=0, std=10, size=(2, 3, 4)),),
    }

    def forward(self, x):
        return torch.abs(x)


@common.parametrize("test_data", Abs.test_parameters)
def test_abs_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](Abs(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Abs.test_parameters)
def test_abs_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](Abs(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Abs.test_parameters)
@common.XfailIfNoCorstone300
def test_abs_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Abs(), test_data(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Abs.test_parameters)
@common.XfailIfNoCorstone320
def test_abs_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Abs(), test_data(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Abs.test_parameters)
@common.SkipIfNoModelConverter
def test_abs_vgf_FP(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Abs(), test_data(), aten_op, exir_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", Abs.test_parameters)
@common.SkipIfNoModelConverter
def test_abs_vgf_INT(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Abs(),
        test_data(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
