# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Tests the rsqrt op.
#

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)


aten_op = "torch.ops.aten.rsqrt.default"
input_t1 = Tuple[torch.Tensor]  # Input x


class Rsqrt(torch.nn.Module):
    test_parameters = {
        "ones_4d": lambda: (torch.ones(1, 10, 10, 10),),
        "rand_4d_1": lambda: (torch.rand(1, 10, 10, 10),),
        "rand_4d_2": lambda: (torch.rand(1, 5, 10, 20),),
        "rand_3d": lambda: (torch.rand(5, 10, 20),),
    }

    def forward(self, x: torch.Tensor):
        return x.rsqrt()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
def test_rsqrt_tosa_MI(test_tensor: torch.Tensor):
    pipeline = TosaPipelineMI[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
def test_rsqrt_tosa_BI(test_tensor: torch.Tensor):
    pipeline = TosaPipelineBI[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.XfailIfNoCorstone300
def test_rsqrt_u55_BI(test_tensor: torch.Tensor):
    pipeline = EthosU55PipelineBI[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_tensor", Rsqrt.test_parameters)
@common.XfailIfNoCorstone320
def test_rsqrt_u85_BI(test_tensor: torch.Tensor):
    pipeline = EthosU85PipelineBI[input_t1](
        Rsqrt(),
        test_tensor(),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()
