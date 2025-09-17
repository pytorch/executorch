# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.common import (
    XfailIfNoCorstone300,
    XfailIfNoCorstone320,
)
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

scalar_input_t = tuple[torch.Tensor, int]


class LshiftScalar(torch.nn.Module):
    torch_op_FP = "torch.ops.aten.__lshift__.Scalar"
    torch_op_INT = "torch.ops.aten.bitwise_left_shift.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_left_shift_Tensor"
    test_data = {
        "randint_neg_8_int8": (
            torch.randint(-8, 8, (1, 12, 3, 4), dtype=torch.int8),
            1,
        ),
        "randint_neg_100_int16": (
            torch.randint(-100, 100, (1, 5, 3, 4), dtype=torch.int16),
            5,
        ),
        "randint_neg_100_int32": (
            torch.randint(-100, 100, (1, 5, 3, 4), dtype=torch.int32),
            2,
        ),
    }

    def forward(self, x: torch.Tensor, shift: int):
        return x << shift


tensor_input_t = tuple[torch.Tensor, torch.Tensor]


class LshiftTensor(torch.nn.Module):
    torch_op = "torch.ops.aten.bitwise_left_shift.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_left_shift_Tensor"
    test_data = {
        "randint_neg_8_tensor_int8": (
            torch.randint(-8, 8, (3, 3), dtype=torch.int8),
            torch.randint(0, 4, (3, 3), dtype=torch.int8),
        ),
        "randint_neg_1024_tensor_int16": (
            torch.randint(-1024, 1024, (3, 3, 3), dtype=torch.int16),
            torch.randint(0, 5, (3, 3, 3), dtype=torch.int16),
        ),
        "randint_0_tensor_int16": (
            torch.randint(0, 127, (1, 2, 3, 3), dtype=torch.int32),
            torch.randint(0, 5, (1, 2, 3, 3), dtype=torch.int32),
        ),
    }

    def forward(self, x: torch.Tensor, shift: torch.Tensor):
        return x.bitwise_left_shift(shift)


##################
## LshiftScalar ##
##################


@common.parametrize("test_data", LshiftScalar.test_data)
def test_bitwise_left_shift_scalar_tosa_FP_scalar(test_data):
    TosaPipelineFP[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_FP,
        LshiftScalar.exir_op,
    ).run()


@common.parametrize("test_data", LshiftScalar.test_data)
def test_bitwise_left_shift_tensor_tosa_INT_scalar(test_data):
    pipeline = TosaPipelineINT[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_INT,
        LshiftScalar.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftScalar.test_data)
@XfailIfNoCorstone300
def test_bitwise_left_shift_tensor_u55_INT_scalar(test_data):
    pipeline = EthosU55PipelineINT[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_INT,
        LshiftScalar.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftScalar.test_data)
@XfailIfNoCorstone320
def test_bitwise_left_shift_tensor_u85_INT_scalar(test_data):
    pipeline = EthosU85PipelineINT[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_INT,
        LshiftScalar.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftScalar.test_data)
@common.SkipIfNoModelConverter
def test_bitwise_left_shift_scalar_vgf_FP_scalar(test_data: scalar_input_t):
    pipeline = VgfPipeline[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_FP,
        LshiftScalar.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", LshiftScalar.test_data)
@common.SkipIfNoModelConverter
def test_bitwise_left_shift_tensor_vgf_INT_scalar(test_data: scalar_input_t):
    pipeline = VgfPipeline[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_INT,
        LshiftScalar.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


##################
## LshiftTensor ##
##################


@common.parametrize("test_data", LshiftTensor.test_data)
def test_bitwise_left_shift_tensor_tosa_FP(test_data):
    TosaPipelineFP[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
    ).run()


@common.parametrize("test_data", LshiftTensor.test_data)
def test_bitwise_left_shift_tensor_tosa_INT(test_data):
    pipeline = TosaPipelineINT[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftTensor.test_data)
@common.XfailIfNoCorstone300
def test_bitwise_left_shift_tensor_u55_INT(test_data):
    pipeline = EthosU55PipelineINT[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftTensor.test_data)
@common.XfailIfNoCorstone320
def test_bitwise_left_shift_tensor_u85_INT(test_data):
    pipeline = EthosU85PipelineINT[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftTensor.test_data)
@common.SkipIfNoModelConverter
def test_bitwise_left_shift_tensor_vgf_FP(test_data: tensor_input_t):
    pipeline = VgfPipeline[tensor_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", LshiftTensor.test_data)
@common.SkipIfNoModelConverter
def test_bitwise_left_shift_tensor_vgf_INT(test_data: tensor_input_t):
    pipeline = VgfPipeline[tensor_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
