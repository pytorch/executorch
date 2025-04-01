# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineBI,
    OpNotSupportedPipeline,
    TosaPipelineBI,
    TosaPipelineMI,
)


input_t = Tuple[torch.Tensor]


class Equal(torch.nn.Module):
    aten_op_BI = "torch.ops.aten.eq.Tensor"
    aten_op_MI = "torch.ops.aten.eq.Scalar"
    exir_op = "executorch_exir_dialects_edge__ops_aten_eq_Tensor"

    def __init__(self, input, other):
        super().__init__()
        self.input_ = input
        self.other_ = other

    def forward(
        self,
        input_: torch.Tensor,
        other_: torch.Tensor | int | float,
    ):
        return input_ == other_

    def get_inputs(self):
        return (self.input_, self.other_)


op_eq_tensor_rank1_ones = Equal(
    torch.ones(5),
    torch.ones(5),
)
op_eq_tensor_rank2_rand = Equal(
    torch.rand(4, 5),
    torch.rand(1, 5),
)
op_eq_tensor_rank3_randn = Equal(
    torch.randn(10, 5, 2),
    torch.randn(10, 5, 2),
)
op_eq_tensor_rank4_randn = Equal(
    torch.randn(3, 2, 2, 2),
    torch.randn(3, 2, 2, 2),
)

op_eq_scalar_rank1_ones = Equal(torch.ones(5), 1.0)
op_eq_scalar_rank2_rand = Equal(torch.rand(4, 5), 0.2)
op_eq_scalar_rank3_randn = Equal(torch.randn(10, 5, 2), -0.1)
op_eq_scalar_rank4_randn = Equal(torch.randn(3, 2, 2, 2), 0.3)

test_data_tensor = {
    "eq_tensor_rank1_ones": op_eq_tensor_rank1_ones,
    "eq_tensor_rank2_rand": op_eq_tensor_rank2_rand,
    "eq_tensor_rank3_randn": op_eq_tensor_rank3_randn,
    "eq_tensor_rank4_randn": op_eq_tensor_rank4_randn,
}

test_data_scalar = {
    "eq_scalar_rank1_ones": op_eq_scalar_rank1_ones,
    "eq_scalar_rank2_rand": op_eq_scalar_rank2_rand,
    "eq_scalar_rank3_randn": op_eq_scalar_rank3_randn,
    "eq_scalar_rank4_randn": op_eq_scalar_rank4_randn,
}


@common.parametrize("test_module", test_data_tensor)
def test_eq_tensor_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module, test_module.get_inputs(), Equal.aten_op_BI, Equal.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_eq_scalar_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module,
        test_module.get_inputs(),
        Equal.aten_op_MI,
        Equal.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor | test_data_scalar)
def test_eq_tosa_BI(test_module):
    pipeline = TosaPipelineBI[input_t](
        test_module, test_module.get_inputs(), Equal.aten_op_BI, Equal.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.XfailIfNoCorstone300
def test_eq_tensor_u55_BI(test_module):
    # EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        "TOSA-0.80+BI+u55",
        {Equal.exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.XfailIfNoCorstone300
def test_eq_scalar_u55_BI(test_module):
    # EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        "TOSA-0.80+BI+u55",
        {Equal.exir_op: 1},
        n_expected_delegates=1,
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_tensor | test_data_scalar,
    xfails={
        "eq_tensor_rank4_randn": "4D fails because boolean Tensors can't be subtracted",
    },
)
@common.XfailIfNoCorstone320
def test_eq_u85_BI(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module,
        test_module.get_inputs(),
        Equal.aten_op_BI,
        Equal.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()
