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
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

scalar_input_t = tuple[torch.Tensor, int]


class LshiftScalar(torch.nn.Module):
    torch_op_MI = "torch.ops.aten.__lshift__.Scalar"
    torch_op_BI = "torch.ops.aten.bitwise_left_shift.Tensor"
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


@common.parametrize("test_data", LshiftScalar.test_data)
def test_lshift_scalar_tosa_MI_scalar(test_data):
    TosaPipelineMI[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_MI,
        LshiftScalar.exir_op,
    ).run()


@common.parametrize("test_data", LshiftScalar.test_data)
def test_bitwise_left_shift_tensor_tosa_BI_scalar(test_data):
    pipeline = TosaPipelineBI[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_BI,
        LshiftScalar.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftScalar.test_data)
@XfailIfNoCorstone300
def test_bitwise_left_shift_tensor_u55_BI_scalar(test_data):
    pipeline = EthosU55PipelineBI[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_BI,
        LshiftScalar.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftScalar.test_data)
@XfailIfNoCorstone320
def test_bitwise_left_shift_tensor_u85_BI_scalar(test_data):
    pipeline = EthosU85PipelineBI[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_BI,
        LshiftScalar.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftTensor.test_data)
def test_lshift_scalar_tosa_MI(test_data):
    TosaPipelineMI[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
    ).run()


@common.parametrize("test_data", LshiftTensor.test_data)
def test_bitwise_left_shift_tensor_tosa_BI(test_data):
    pipeline = TosaPipelineBI[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftTensor.test_data)
@XfailIfNoCorstone300
def test_bitwise_left_shift_tensor_u55_BI(test_data):
    pipeline = EthosU55PipelineBI[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", LshiftTensor.test_data)
@XfailIfNoCorstone320
def test_bitwise_left_shift_tensor_u85_BI(test_data):
    pipeline = EthosU85PipelineBI[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
