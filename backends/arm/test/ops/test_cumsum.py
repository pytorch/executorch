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

input_t1 = Tuple[torch.Tensor, int]
aten_op = "torch.ops.aten.cumsum.default"

"""
Tests the aten.cumsum operator by decomposing it into a convolution and
verifying results across various dims and pipelines.
"""


class CumsumModule(torch.nn.Module):
    test_parameters = {
        "1d_dim0": lambda: (torch.rand(10), 0),
        "1d_dim_neg1": lambda: (torch.rand(10), -1),
        "2d_dim1": lambda: (torch.rand(5, 6), 1),
        "3d_dim2": lambda: (torch.rand(2, 3, 4), 2),
        "3d_dim0": lambda: (torch.rand(2, 3, 4), 0),
        "4d_dim3": lambda: (torch.rand(1, 2, 3, 4), 3),
        "4d_dim1": lambda: (torch.rand(1, 2, 3, 4), 1),
    }

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cumsum(x, dim)


@common.parametrize("test_data", CumsumModule.test_parameters)
def test_cumsum_tosa_FP(test_data: input_t1):
    module = CumsumModule()
    args = test_data()
    pipeline = TosaPipelineFP[input_t1](
        module,
        args,
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", CumsumModule.test_parameters)
def test_cumsum_tosa_INT(test_data: input_t1):
    module = CumsumModule()
    args = test_data()
    pipeline = TosaPipelineINT[input_t1](
        module,
        args,
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", CumsumModule.test_parameters)
@common.SkipIfNoModelConverter
def test_cumsum_vgf_no_quant(test_data: input_t1):
    module = CumsumModule()
    args = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        args,
        aten_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", CumsumModule.test_parameters)
@common.SkipIfNoModelConverter
def test_cumsum_vgf_quant(test_data: input_t1):
    module = CumsumModule()
    args = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        args,
        aten_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", CumsumModule.test_parameters)
@common.XfailIfNoCorstone300
def test_cumsum_u55_INT(test_data: input_t1):
    module = CumsumModule()
    args = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        module,
        args,
        aten_ops=aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", CumsumModule.test_parameters)
@common.XfailIfNoCorstone320
def test_cumsum_u85_INT(test_data: input_t1):
    module = CumsumModule()
    args = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        module,
        args,
        aten_ops=aten_op,
        exir_ops=[],
    )
    pipeline.run()
