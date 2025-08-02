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
    "zeros": lambda: (torch.zeros(1, 10, 10, 10)),
    "ones": lambda: (torch.ones(10, 10, 10)),
    "rand": lambda: (torch.rand(10, 10) - 0.5),
    "randn_pos": lambda: (torch.randn(10) + 10),
    "randn_neg": lambda: (torch.randn(10) - 10),
    "ramp": lambda: (torch.arange(-16, 16, 0.2)),
}

aten_op = "torch.ops.aten.hardtanh.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_hardtanh_default"

input_t = Tuple[torch.Tensor]


class HardTanh(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.hardTanh = torch.nn.Hardtanh()

    def forward(self, x):
        return self.hardTanh(x)


@common.parametrize("test_data", test_data_suite)
def test_hardtanh_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t](HardTanh(), (test_data(),), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_hardtanh_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t](
        HardTanh(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_hardtanh_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t](
        HardTanh(),
        (test_data(),),
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_hardtanh_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t](
        HardTanh(),
        (test_data(),),
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_hardtanh_vgf_FP(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t](
        HardTanh(), (test_data(),), aten_op, exir_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_hardtanh_vgf_INT(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t](
        HardTanh(),
        (test_data(),),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
