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

aten_op = "torch.ops.aten.atanh.default"
exir_op = "executorch_exir_dialects_edge__ops_aten__atanh_default"


input_t1 = Tuple[torch.Tensor]


test_data_suite = {
    "zeros": torch.zeros(1, 10, 10, 10),
    "zeros_alt_shape": torch.zeros(1, 10, 3, 5),
    "rand": torch.rand(10, 10) - 0.5,
    "rand_alt_shape": torch.rand(1, 10, 3, 5) - 0.5,
    "ramp": torch.arange(-1, 1, 0.2),
    "near_bounds": torch.tensor([-0.99, -0.9, 0.9, 0.99]),
    "on_bounds": torch.tensor([-1.0, 1.0]),
}


class Atanh(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.atanh(x)


@common.parametrize("test_data", test_data_suite)
def test_atanh_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        Atanh(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_atanh_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        Atanh(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    if torch.any(test_data >= 1) or torch.any(test_data <= -1):
        # The quantized model will saturate to max/min values while the
        # original model will return inf/-inf, so comparison wont be valid here.
        pipeline.pop_stage("run_method_and_compare_outputs.original_model")
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_data", test_data_suite)
def test_atanh_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Atanh(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_atanh_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Atanh(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_atanh_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Atanh(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_atanh_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Atanh(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
        quantize=True,
    )
    pipeline.run()
