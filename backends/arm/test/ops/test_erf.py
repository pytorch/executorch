# Copyright 2025-2026 Arm Limited and/or its affiliates.
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

aten_op = "torch.ops.aten.erf.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_erf_default"
input_t1 = Tuple[torch.Tensor]  # Input x


class Erf(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.erf(x)

    test_data: dict[str, input_t1] = {
        "zeros": lambda: (torch.zeros(1, 10, 10, 10),),
        "ones": lambda: (torch.ones(10, 10, 10),),
        "rand": lambda: ((torch.rand(10, 10) - 0.5),),
        "randn_pos": lambda: ((torch.randn(1, 4, 4, 4) + 10),),
        "randn_neg": lambda: ((torch.randn(1, 4, 4, 4) - 10),),
        "ramp": lambda: (torch.arange(-16, 16, 0.2),),
    }

    test_data_bf16: dict[str, input_t1] = {
        "rand_bf16": lambda: ((torch.rand(8, 8, dtype=torch.bfloat16) - 0.5),),
        "ramp_bf16": lambda: (torch.arange(-8, 8, 0.5, dtype=torch.bfloat16),),
    }


@common.parametrize("test_data", Erf.test_data | Erf.test_data_bf16)
def test_erf_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        Erf(), test_data(), aten_op, exir_op, tosa_extensions=["bf16"]
    )
    pipeline.run()


@common.parametrize("test_data", Erf.test_data)
def test_erf_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](Erf(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Erf.test_data)
@common.XfailIfNoCorstone300
def test_erf_u55_INT(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        Erf(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Erf.test_data)
@common.XfailIfNoCorstone320
def test_erf_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        Erf(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Erf.test_data)
@common.SkipIfNoModelConverter
def test_erf_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Erf(),
        test_data(),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", Erf.test_data)
@common.SkipIfNoModelConverter
def test_erf_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Erf(),
        test_data(),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
