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


class LessEqual(torch.nn.Module):
    aten_op_tensor = "torch.ops.aten.le.Tensor"
    aten_op_scalar = "torch.ops.aten.le.Scalar"
    exir_op = "executorch_exir_dialects_edge__ops_aten_le_Tensor"

    def __init__(self, input, other):
        super().__init__()
        self.input_ = input
        self.other_ = other

    def forward(
        self,
        input_: torch.Tensor,
        other_: torch.Tensor,
    ):
        return input_ <= other_

    def get_inputs(self):
        return (self.input_, self.other_)


op_le_tensor_rank1_ones = LessEqual(
    torch.ones(5),
    torch.ones(5),
)
op_le_tensor_rank2_rand = LessEqual(
    torch.rand(4, 5),
    torch.rand(1, 5),
)
op_le_tensor_rank3_randn = LessEqual(
    torch.randn(10, 5, 2),
    torch.randn(10, 5, 2),
)
op_le_tensor_rank4_randn = LessEqual(
    torch.randn(3, 2, 2, 2),
    torch.randn(3, 2, 2, 2),
)

op_le_scalar_rank1_ones = LessEqual(torch.ones(5), 1.0)
op_le_scalar_rank2_rand = LessEqual(torch.rand(4, 5), 0.2)
op_le_scalar_rank3_randn = LessEqual(torch.randn(10, 5, 2), -0.1)
op_le_scalar_rank4_randn = LessEqual(torch.randn(3, 2, 2, 2), 0.3)

test_data_tensor = {
    "le_tensor_rank1_ones": lambda: op_le_tensor_rank1_ones,
    "le_tensor_rank2_rand": lambda: op_le_tensor_rank2_rand,
    "le_tensor_rank3_randn": lambda: op_le_tensor_rank3_randn,
    "le_tensor_rank4_randn": lambda: op_le_tensor_rank4_randn,
}

test_data_scalar = {
    "le_scalar_rank1_ones": lambda: op_le_scalar_rank1_ones,
    "le_scalar_rank2_rand": lambda: op_le_scalar_rank2_rand,
    "le_scalar_rank3_randn": lambda: op_le_scalar_rank3_randn,
    "le_scalar_rank4_randn": lambda: op_le_scalar_rank4_randn,
}


@common.parametrize("test_module", test_data_tensor)
def test_le_tensor_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module(),
        test_module().get_inputs(),
        LessEqual.aten_op_tensor,
        LessEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_le_scalar_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module(),
        test_module().get_inputs(),
        LessEqual.aten_op_scalar,
        LessEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
def test_le_tensor_tosa_BI(test_module):
    pipeline = TosaPipelineBI[input_t](
        test_module(),
        test_module().get_inputs(),
        LessEqual.aten_op_tensor,
        LessEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_le_scalar_tosa_BI(test_module):
    pipeline = TosaPipelineBI[input_t](
        test_module(),
        test_module().get_inputs(),
        LessEqual.aten_op_tensor,
        LessEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.XfailIfNoCorstone300
def test_le_tensor_u55_BI_not_delegated(test_module):
    # GREATER_EQUAL is not supported on U55. LE uses the GREATER_EQUAL Tosa operator.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {LessEqual.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.XfailIfNoCorstone300
def test_le_scalar_u55_BI_not_delegated(test_module):
    # GREATER_EQUAL is not supported on U55. LE uses the GREATER_EQUAL Tosa operator.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {LessEqual.exir_op: 1},
        n_expected_delegates=1,
        quantize=True,
        u55_subset=True,
    )
    pipeline.dump_operator_distribution("export")
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_tensor,
    xfails={
        "le_tensor_rank4_randn": "4D fails because boolean Tensors can't be subtracted"
    },
)
@common.XfailIfNoCorstone320
def test_le_tensor_u85_BI(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module(),
        test_module().get_inputs(),
        LessEqual.aten_op_tensor,
        LessEqual.exir_op,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_scalar,
    xfails={
        "le_scalar_rank4_randn": "4D fails because boolean Tensors can't be subtracted"
    },
)
@common.XfailIfNoCorstone320
def test_le_scalar_u85_BI(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module(),
        test_module().get_inputs(),
        LessEqual.aten_op_tensor,
        LessEqual.exir_op,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
