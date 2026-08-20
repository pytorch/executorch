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
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.log10.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_log10_default"

input_t1 = Tuple[torch.Tensor]


def _tensor(values):
    return torch.tensor(values, dtype=torch.float32)


test_data_suite = {
    # (test_name, test_data)
    "tiny_positive": lambda: (_tensor([5e-4, 8e-4, 9e-4, 1e-3, 1.2e-3])),
    "mixed_range": lambda: (_tensor([1e-4, 5e-4, 2e-3, 1e-2, 5e-2])),
    "ones_rank4": lambda: (torch.ones(1, 10, 10, 10)),
    "ones_rank3": lambda: (torch.ones(10, 10, 10)),
    "rand": lambda: (torch.rand(10, 10) + 0.001),
    "randn_pos": lambda: (torch.randn(10) + 10),
    "randn_spread": lambda: (torch.max(torch.Tensor([0.1]), torch.randn(10) * 100)),
    "ramp": lambda: (torch.arange(0.01, 20, 0.2)),
}


class Log10(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log10(x)


@common.parametrize("test_data", test_data_suite)
def test_log10_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](Log10(), (test_data(),), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_log10_u55_INT(test_data: input_t1):
    EthosU55PipelineINT[input_t1](
        Log10(),
        (test_data(),),
        aten_op,
        exir_op,
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_log10_u85_INT(test_data: input_t1):
    EthosU85PipelineINT[input_t1](
        Log10(),
        (test_data(),),
        aten_op,
        exir_op,
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_log10_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Log10(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
