# Copyright 2026 Arm Limited and/or its affiliates.
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

aten_op = "torch.ops.aten.erfinv.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_erfinv_default"

input_t1 = Tuple[torch.Tensor]

test_data_suite = {
    "zeros": torch.zeros(1, 10, 10, 10),
    "small": torch.randn(100) * 0.01,
    "mid": torch.rand(10, 10) * 1.8 - 0.9,
    "near_pos_bound": torch.full((32,), 0.99),
    "near_neg_bound": torch.full((32,), -0.99),
    "pos_one": torch.full((32,), 1.0),
    "neg_one": torch.full((32,), -1.0),
    "ramp": torch.arange(-0.99, 0.99, 0.02),
}


test_data_nan_outputs = {
    "pos_two": torch.full((32,), 2.0),
    "neg_two": torch.full((32,), -2.0),
}


test_data_fp16 = {
    "rand_fp16": (torch.rand(8, 8, dtype=torch.float16) * 1.8 - 0.9),
    "ramp_fp16": torch.arange(-0.9, 0.9, 0.1, dtype=torch.float16),
}


class Erfinv(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.erfinv(x)


@common.parametrize(
    "test_data", test_data_suite | test_data_nan_outputs | test_data_fp16
)
def test_erfinv_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Erfinv(),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_erfinv_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](Erfinv(), (test_data,), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_erfinv_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Erfinv(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_erfinv_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Erfinv(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data", test_data_suite | test_data_nan_outputs | test_data_fp16
)
@common.SkipIfNoModelConverter
def test_erfinv_vgf_no_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Erfinv(),
        (test_data,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_erfinv_vgf_quant(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Erfinv(),
        (test_data,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
