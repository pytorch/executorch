# Copyright 2025 Arm Limited and/or its affiliates.
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


@common.parametrize("test_data", Erf.test_data)
def test_erf_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](Erf(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Erf.test_data)
def test_erf_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](Erf(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", Erf.test_data)
@common.XfailIfNoCorstone300
def test_erf_u55_BI(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        Erf(), test_data(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_data", Erf.test_data)
@common.XfailIfNoCorstone320
def test_erf_u85_BI(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        Erf(), test_data(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()
