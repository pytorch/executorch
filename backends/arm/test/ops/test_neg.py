# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor]


class Neg(torch.nn.Module):

    aten_op = "torch.ops.aten.neg.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_neg_default"

    test_data: Dict[str, input_t1] = {
        "rank_1_ramp": (torch.arange(-16, 16, 0.2),),
        "rank_2_rand_uniform": (torch.rand(10, 10) - 0.5,),
        "rank_3_all_ones": (torch.ones(10, 10, 10),),
        "rank_4_all_zeros": (torch.zeros(1, 10, 10, 10),),
        "rank_4_randn_pos": (torch.randn(1, 4, 4, 4) + 10,),
        "rank_4_randn_neg": (torch.randn(1, 4, 4, 4) - 10,),
    }

    def forward(self, x: torch.Tensor):
        return torch.neg(x)


@common.parametrize("test_data", Neg.test_data)
def test_neg_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](Neg(), test_data, Neg.aten_op, Neg.exir_op)
    pipeline.run()


@common.parametrize("test_data", Neg.test_data)
def test_neg_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](Neg(), test_data, Neg.aten_op, Neg.exir_op)
    pipeline.run()


@common.parametrize("test_data", Neg.test_data)
@common.XfailIfNoCorstone300
def test_neg_u55_INT(test_data: input_t1):
    pipeline = EthosU55PipelineINT[input_t1](
        Neg(),
        test_data,
        Neg.aten_op,
        Neg.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Neg.test_data)
@common.XfailIfNoCorstone320
def test_neg_u85_INT(test_data: input_t1):
    pipeline = EthosU85PipelineINT[input_t1](
        Neg(),
        test_data,
        Neg.aten_op,
        Neg.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Neg.test_data)
@common.SkipIfNoModelConverter
def test_neg_vgf_FP(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Neg(), test_data, Neg.aten_op, Neg.exir_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", Neg.test_data)
@common.SkipIfNoModelConverter
def test_neg_vgf_INT(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        Neg(),
        test_data,
        Neg.aten_op,
        Neg.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
