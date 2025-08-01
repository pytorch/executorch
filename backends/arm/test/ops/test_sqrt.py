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
)


class Sqrt(torch.nn.Module):
    input_t = Tuple[torch.Tensor]
    aten_op_FP = "torch.ops.aten.sqrt.default"
    exir_op_FP = "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Tensor"

    aten_op_INT = "torch.ops.aten.pow.Tensor_Scalar"
    exir_op_INT = "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sqrt(x)

    test_data: Dict[str, input_t] = {
        "sqrt_tensor_rank1_ones": lambda: (torch.ones(10),),
        "sqrt_tensor_rank2_random": lambda: (torch.rand(5, 10),),
        "sqrt_tensor_rank3_ones": lambda: (torch.ones(2, 3, 4),),
        "sqrt_tensor_rank4_random": lambda: (torch.rand(1, 3, 8, 8),),
        "sqrt_tensor_rank4_multibatch": lambda: (torch.rand(2, 3, 4, 4),),
    }


fvp_xfails = {
    "sqrt_tensor_rank4_multibatch": "MLETORCH-517 : Multiple batches not supported",
}


@common.parametrize("test_data", Sqrt.test_data)
def test_sqrt_tosa_FP(test_data: Sqrt.input_t):
    pipeline = TosaPipelineFP[Sqrt.input_t](
        Sqrt(),
        test_data(),
        Sqrt.aten_op_FP,
        Sqrt.exir_op_FP,
    )
    pipeline.run()


@common.parametrize("test_data", Sqrt.test_data)
def test_sqrt_tosa_INT(test_data: Sqrt.input_t):
    pipeline = TosaPipelineINT[Sqrt.input_t](
        Sqrt(),
        test_data(),
        Sqrt.aten_op_INT,
        Sqrt.exir_op_INT,
    )
    pipeline.run()


@common.parametrize("test_data", Sqrt.test_data, fvp_xfails)
@common.XfailIfNoCorstone300
def test_sqrt_u55_INT(test_data: Sqrt.input_t):
    pipeline = EthosU55PipelineINT[Sqrt.input_t](
        Sqrt(),
        test_data(),
        Sqrt.aten_op_INT,
        Sqrt.exir_op_INT,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Sqrt.test_data, fvp_xfails)
@common.XfailIfNoCorstone320
def test_sqrt_u85_INT(test_data: Sqrt.input_t):
    pipeline = EthosU85PipelineINT[Sqrt.input_t](
        Sqrt(),
        test_data(),
        Sqrt.aten_op_INT,
        Sqrt.exir_op_INT,
        run_on_fvp=True,
    )
    pipeline.run()
