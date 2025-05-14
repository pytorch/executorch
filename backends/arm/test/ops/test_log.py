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


class Log(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)


@common.parametrize("test_data", test_data_suite)
def test_log_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](Log(), (test_data(),), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_log_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](Log(), (test_data(),), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_log_u55_BI(test_data: input_t1):
    EthosU55PipelineBI[input_t1](
        Log(),
        (test_data(),),
        aten_op,
        exir_op,
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_log_u85_BI(test_data: input_t1):
    EthosU85PipelineBI[input_t1](
        Log(),
        (test_data(),),
        aten_op,
        exir_op,
        run_on_fvp=True,
    ).run()
