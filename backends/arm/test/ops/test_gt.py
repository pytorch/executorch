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


class Greater(torch.nn.Module):
    aten_op_tensor = "torch.ops.aten.gt.Tensor"
    aten_op_scalar = "torch.ops.aten.gt.Scalar"
    exir_op = "executorch_exir_dialects_edge__ops_aten_gt_Tensor"

    def __init__(self, input, other):
        super().__init__()
        self.input_ = input
        self.other_ = other

    def forward(
        self,
        input_: torch.Tensor,
        other_: torch.Tensor | int | float,
    ):
        return input_ > other_

    def get_inputs(self):
        return (self.input_, self.other_)


op_gt_tensor_rank1_ones = Greater(
    torch.ones(5),
    torch.ones(5),
)
op_gt_tensor_rank2_rand = Greater(
    torch.rand(4, 5),
    torch.rand(1, 5),
)
op_gt_tensor_rank3_randn = Greater(
    torch.randn(10, 5, 2),
    torch.randn(10, 5, 2),
)
op_gt_tensor_rank4_randn = Greater(
    torch.randn(3, 2, 2, 2),
    torch.randn(3, 2, 2, 2),
)

op_gt_scalar_rank1_ones = Greater(torch.ones(5), 1.0)
op_gt_scalar_rank2_rand = Greater(torch.rand(4, 5), 0.2)
op_gt_scalar_rank3_randn = Greater(torch.randn(10, 5, 2), -0.1)
op_gt_scalar_rank4_randn = Greater(torch.randn(3, 2, 2, 2), 0.3)

test_data_tensor = {
    "gt_tensor_rank1_ones": op_gt_tensor_rank1_ones,
    "gt_tensor_rank2_rand": op_gt_tensor_rank2_rand,
    "gt_tensor_rank3_randn": op_gt_tensor_rank3_randn,
    "gt_tensor_rank4_randn": op_gt_tensor_rank4_randn,
}

test_data_scalar = {
    "gt_scalar_rank1_ones": op_gt_scalar_rank1_ones,
    "gt_scalar_rank2_rand": op_gt_scalar_rank2_rand,
    "gt_scalar_rank3_randn": op_gt_scalar_rank3_randn,
    "gt_scalar_rank4_randn": op_gt_scalar_rank4_randn,
}


@common.parametrize("test_module", test_data_tensor)
def test_gt_tensor_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module, test_module.get_inputs(), Greater.aten_op_tensor, Greater.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_gt_scalar_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module, test_module.get_inputs(), Greater.aten_op_scalar, Greater.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
def test_gt_tensor_tosa_BI(test_module):
    pipeline = TosaPipelineBI[input_t](
        test_module, test_module.get_inputs(), Greater.aten_op_tensor, Greater.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_gt_scalar_tosa_BI(test_module):
    pipeline = TosaPipelineBI[input_t](
        test_module, test_module.get_inputs(), Greater.aten_op_tensor, Greater.exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.XfailIfNoCorstone300
def test_gt_tensor_u55_BI(test_module):
    # Greater is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        "TOSA-0.80+BI+u55",
        {Greater.exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.XfailIfNoCorstone300
def test_gt_scalar_u55_BI(test_module):
    # Greater is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module,
        test_module.get_inputs(),
        "TOSA-0.80+BI+u55",
        {Greater.exir_op: 1},
        n_expected_delegates=1,
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_tensor,
    xfails={
        "gt_tensor_rank4_randn": "MLETORCH-847: Boolean eq result unstable on U85",
    },
)
@common.XfailIfNoCorstone320
def test_gt_tensor_u85_BI(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module,
        test_module.get_inputs(),
        Greater.aten_op_tensor,
        Greater.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_scalar,
    xfails={
        "gt_scalar_rank4_randn": "MLETORCH-847: Boolean eq result unstable on U85",
    },
)
@common.XfailIfNoCorstone320
def test_gt_scalar_u85_BI(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module,
        test_module.get_inputs(),
        Greater.aten_op_tensor,
        Greater.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()
