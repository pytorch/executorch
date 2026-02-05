# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
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

aten_op = "torch.ops.aten.log.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_log_default"

input_t1 = Tuple[torch.Tensor]

test_data_suite = {
    # (test_name, test_data)
    "ones_rank4": lambda: (torch.ones(1, 10, 10, 10)),
    "ones_rank3": lambda: (torch.ones(10, 10, 10)),
    "rand": lambda: (torch.rand(10, 10) + 0.001),
    "randn_pos": lambda: (torch.randn(10) + 10),
    "randn_spread": lambda: (torch.max(torch.Tensor([0.0]), torch.randn(10) * 100)),
    "ramp": lambda: (torch.arange(0.01, 20, 0.2)),
}

test_data_suite_bf16 = {
    "rand_bf16": lambda: (torch.rand(6, 6, dtype=torch.bfloat16) + 1),
    "ramp_bf16": lambda: (torch.arange(0.5, 8, 0.5, dtype=torch.bfloat16)),
}


class Log(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)


@common.parametrize("test_data", test_data_suite | test_data_suite_bf16)
def test_log_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        Log(), (test_data(),), aten_op, exir_op, tosa_extensions=["bf16"]
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_log_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](Log(), (test_data(),), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_log_u55_INT(test_data: input_t1):
    EthosU55PipelineINT[input_t1](
        Log(),
        (test_data(),),
        aten_op,
        exir_op,
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_log_u85_INT(test_data: input_t1):
    EthosU85PipelineINT[input_t1](
        Log(),
        (test_data(),),
        aten_op,
        exir_op,
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_log_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Log(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_log_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Log(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
