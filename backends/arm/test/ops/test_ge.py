# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t = Tuple[torch.Tensor]


class GreaterEqual(torch.nn.Module):
    aten_op_tensor = "torch.ops.aten.ge.Tensor"
    aten_op_scalar = "torch.ops.aten.ge.Scalar"
    exir_op = "executorch_exir_dialects_edge__ops_aten_ge_Tensor"

    def __init__(self, input, other):
        super().__init__()
        self.input_ = input
        self.other_ = other

    def forward(
        self,
        input_: torch.Tensor,
        other_: torch.Tensor | int | float,
    ):
        return input_ >= other_

    def get_inputs(self):
        return (self.input_, self.other_)


op_ge_tensor_rank1_ones = GreaterEqual(
    torch.ones(5),
    torch.ones(5),
)
op_ge_tensor_rank2_rand = GreaterEqual(
    torch.rand(4, 5),
    torch.rand(1, 5),
)
op_ge_tensor_rank3_randn = GreaterEqual(
    torch.randn(10, 5, 2),
    torch.randn(10, 5, 2),
)
op_ge_tensor_rank4_randn = GreaterEqual(
    torch.randn(3, 2, 2, 2),
    torch.randn(3, 2, 2, 2),
)

op_ge_scalar_rank1_ones = GreaterEqual(torch.ones(5), 1.0)
op_ge_scalar_rank2_rand = GreaterEqual(torch.rand(4, 5), 0.2)
op_ge_scalar_rank3_randn = GreaterEqual(torch.randn(10, 5, 2), -0.1)
op_ge_scalar_rank4_randn = GreaterEqual(torch.randn(3, 2, 2, 2), 0.3)

test_data_tensor = {
    "ge_tensor_rank1_ones": lambda: op_ge_tensor_rank1_ones,
    "ge_tensor_rank2_rand": lambda: op_ge_tensor_rank2_rand,
    "ge_tensor_rank3_randn": lambda: op_ge_tensor_rank3_randn,
    "ge_tensor_rank4_randn": lambda: op_ge_tensor_rank4_randn,
}

test_data_scalar = {
    "ge_scalar_rank1_ones": lambda: op_ge_scalar_rank1_ones,
    "ge_scalar_rank2_rand": lambda: op_ge_scalar_rank2_rand,
    "ge_scalar_rank3_randn": lambda: op_ge_scalar_rank3_randn,
    "ge_scalar_rank4_randn": lambda: op_ge_scalar_rank4_randn,
}


@common.parametrize("test_module", test_data_tensor)
def test_ge_tensor_tosa_FP(test_module):
    pipeline = TosaPipelineFP[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_tensor,
        GreaterEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_ge_scalar_tosa_FP(test_module):
    pipeline = TosaPipelineFP[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_scalar,
        GreaterEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
def test_ge_tensor_tosa_INT(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_tensor,
        GreaterEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_ge_scalar_tosa_INT(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_tensor,
        GreaterEqual.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.XfailIfNoCorstone300
def test_ge_tensor_u55_INT(test_module):
    # GREATER_EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {GreaterEqual.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.XfailIfNoCorstone300
def test_ge_scalar_u55_INT(test_module):
    # GREATER_EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {GreaterEqual.exir_op: 1},
        n_expected_delegates=1,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_tensor,
)
@common.XfailIfNoCorstone320
def test_ge_tensor_u85_INT(test_module):
    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_tensor,
        GreaterEqual.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize(
    "test_module",
    test_data_scalar,
)
@common.XfailIfNoCorstone320
def test_ge_scalar_u85_INT(test_module):
    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_tensor,
        GreaterEqual.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.SkipIfNoModelConverter
def test_ge_tensor_vgf_FP(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_tensor,
        GreaterEqual.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.SkipIfNoModelConverter
def test_ge_tensor_vgf_INT(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_tensor,
        GreaterEqual.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.SkipIfNoModelConverter
def test_ge_scalar_vgf_FP(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_scalar,
        GreaterEqual.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.SkipIfNoModelConverter
def test_ge_scalar_vgf_INT(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        GreaterEqual.aten_op_tensor,
        GreaterEqual.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
