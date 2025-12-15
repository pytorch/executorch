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


class Equal(torch.nn.Module):
    aten_op_Tensor = "torch.ops.aten.eq.Tensor"
    aten_op_Scalar = "torch.ops.aten.eq.Scalar"
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
    "eq_tensor_rank1_ones": lambda: op_eq_tensor_rank1_ones,
    "eq_tensor_rank2_rand": lambda: op_eq_tensor_rank2_rand,
    "eq_tensor_rank3_randn": lambda: op_eq_tensor_rank3_randn,
    "eq_tensor_rank4_randn": lambda: op_eq_tensor_rank4_randn,
}

test_data_scalar = {
    "eq_scalar_rank1_ones": lambda: op_eq_scalar_rank1_ones,
    "eq_scalar_rank2_rand": lambda: op_eq_scalar_rank2_rand,
    "eq_scalar_rank3_randn": lambda: op_eq_scalar_rank3_randn,
    "eq_scalar_rank4_randn": lambda: op_eq_scalar_rank4_randn,
}


@common.parametrize("test_module", test_data_tensor)
def test_eq_scalar_tosa_FP_tensor(test_module):
    pipeline = TosaPipelineFP[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_eq_scalar_tosa_FP(test_module):
    pipeline = TosaPipelineFP[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Scalar,
        Equal.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
def test_eq_scalar_tosa_INT_tensor(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_eq_scalar_tosa_INT(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
def test_eq_tensor_16a8w_tosa_INT(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
def test_eq_scalar_tosa_INT_16a8w(test_module):
    pipeline = TosaPipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.XfailIfNoCorstone300
def test_eq_scalar_u55_INT_tensor(test_module):
    # EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {Equal.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.XfailIfNoCorstone300
def test_eq_scalar_u55_INT(test_module):
    # EQUAL is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        {Equal.exir_op: 1},
        n_expected_delegates=1,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.XfailIfNoCorstone320
def test_eq_scalar_u85_INT_tensor(test_module):
    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.XfailIfNoCorstone320
def test_eq_scalar_u85_INT(test_module):
    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.XfailIfNoCorstone320
def test_eq_tensor_16a8w_u85_INT(test_module):
    """Test eq operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
        per_channel_quantization=per_channel_quantization,
        a16w8_quantization=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.XfailIfNoCorstone320
def test_eq_scalar_16a8w_u85_INT(test_module):
    """Test eq operation (scalar) with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
        per_channel_quantization=per_channel_quantization,
        a16w8_quantization=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.SkipIfNoModelConverter
def test_eq_scalar_vgf_no_quant_tensor(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.SkipIfNoModelConverter
def test_eq_scalar_vgf_no_quant(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Scalar,
        Equal.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_tensor)
@common.SkipIfNoModelConverter
def test_eq_scalar_vgf_quant_tensor(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_scalar)
@common.SkipIfNoModelConverter
def test_eq_scalar_vgf_quant(test_module):
    pipeline = VgfPipeline[input_t](
        test_module(),
        test_module().get_inputs(),
        Equal.aten_op_Tensor,
        Equal.exir_op,
        quantize=True,
    )
    pipeline.run()
