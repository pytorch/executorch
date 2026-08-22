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

aten_op = "torch.ops.aten.fill_.Scalar"
exir_op = "executorch_exir_dialects_edge__ops_aten_full_like_default"

input_t1 = Tuple[torch.Tensor]

test_data_suite = {
    "ones_float": [torch.ones(2, 3), 5.0],
    "ones_int": [torch.ones(2, 3), -3],
}


class FillScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor, fill_value: int | float):
        mask = torch.full_like(y, 0)
        mask.fill_(fill_value)
        return mask * y


@common.parametrize("test_data", test_data_suite)
def test_fill_scalar_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        FillScalar(),
        (*test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_fill_scalar_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        FillScalar(),
        (*test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_fill_scalar_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        FillScalar(),
        (*test_data,),
        aten_ops=[aten_op],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_fill_scalar_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        FillScalar(),
        (*test_data,),
        aten_ops=[aten_op],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_fill_scalar_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        FillScalar(),
        (*test_data,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_fill_scalar_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        FillScalar(),
        (*test_data,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
