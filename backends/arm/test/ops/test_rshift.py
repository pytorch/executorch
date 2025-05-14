# Copyright 2024-2025 Arm Limited and/or its affiliates.
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


class RshiftScalar(torch.nn.Module):
    torch_op_MI = "torch.ops.aten.__rshift__.Scalar"
    torch_op_BI = "torch.ops.aten.bitwise_right_shift.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_right_shift_Tensor"
    test_data = {
        "randint_neg_100_int8": lambda: (
            torch.randint(-100, 100, (1, 12, 3, 4), dtype=torch.int8),
            1,
        ),
        "randint_neg_100_int16": lambda: (
            torch.randint(-100, 100, (1, 5, 3, 4), dtype=torch.int16),
            5,
        ),
        "randint_neg_100_int32": lambda: (
            torch.randint(-100, 100, (1, 5, 3, 4), dtype=torch.int32),
            2,
        ),
    }

    def forward(self, x: torch.Tensor, shift: int):
        return x >> shift


tensor_input_t = tuple[torch.Tensor, torch.Tensor]


class RshiftTensor(torch.nn.Module):
    torch_op = "torch.ops.aten.bitwise_right_shift.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_right_shift_Tensor"
    test_data = {
        "randint_neg_128_int8": lambda: (
            torch.randint(-128, 127, (3, 3), dtype=torch.int8),
            torch.randint(0, 5, (3, 3), dtype=torch.int8),
        ),
        "randint_neg_1024_int16": lambda: (
            torch.randint(-1024, 1024, (3, 3, 3), dtype=torch.int16),
            torch.randint(0, 5, (3, 3, 3), dtype=torch.int16),
        ),
        "randint_0_127_int32": lambda: (
            torch.randint(0, 127, (1, 2, 3, 3), dtype=torch.int32),
            torch.randint(0, 5, (1, 2, 3, 3), dtype=torch.int32),
        ),
    }

    def forward(self, x: torch.Tensor, shift: torch.Tensor):
        return x.bitwise_right_shift(shift)


@common.parametrize("test_data", RshiftScalar.test_data)
def test_rshift_scalar_tosa_MI_scalar(test_data):
    TosaPipelineMI[scalar_input_t](
        RshiftScalar(),
        test_data(),
        RshiftScalar.torch_op_MI,
        RshiftScalar.exir_op,
    ).run()


@common.parametrize("test_data", RshiftScalar.test_data)
def test_bitwise_right_shift_tensor_tosa_BI_scalar(test_data):
    pipeline = TosaPipelineBI[scalar_input_t](
        RshiftScalar(),
        test_data(),
        RshiftScalar.torch_op_BI,
        RshiftScalar.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", RshiftScalar.test_data)
@XfailIfNoCorstone300
def test_bitwise_right_shift_tensor_u55_BI_scalar(test_data):
    pipeline = EthosU55PipelineBI[scalar_input_t](
        RshiftScalar(),
        test_data(),
        RshiftScalar.torch_op_BI,
        RshiftScalar.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")

    # Forced rounding in U55 HW causes off-by-one errors.
    pipeline.change_args("run_method_and_compare_outputs", inputs=test_data(), atol=1)
    pipeline.run()


@common.parametrize("test_data", RshiftScalar.test_data)
@XfailIfNoCorstone320
def test_bitwise_right_shift_tensor_u85_BI_scalar(test_data):
    pipeline = EthosU85PipelineBI[scalar_input_t](
        RshiftScalar(),
        test_data(),
        RshiftScalar.torch_op_BI,
        RshiftScalar.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", RshiftTensor.test_data)
def test_rshift_scalar_tosa_MI(test_data):
    TosaPipelineMI[scalar_input_t](
        RshiftTensor(),
        test_data(),
        RshiftTensor.torch_op,
        RshiftTensor.exir_op,
    ).run()


@common.parametrize("test_data", RshiftTensor.test_data)
def test_bitwise_right_shift_tensor_tosa_BI(test_data):
    pipeline = TosaPipelineBI[scalar_input_t](
        RshiftTensor(),
        test_data(),
        RshiftTensor.torch_op,
        RshiftTensor.exir_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", RshiftTensor.test_data)
@XfailIfNoCorstone300
def test_bitwise_right_shift_tensor_u55_BI(test_data):
    pipeline = EthosU55PipelineBI[scalar_input_t](
        RshiftTensor(),
        test_data(),
        RshiftTensor.torch_op,
        RshiftTensor.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")

    # Forced rounding in U55 HW causes off-by-one errors.
    pipeline.change_args("run_method_and_compare_outputs", inputs=test_data(), atol=1)
    pipeline.run()


@common.parametrize("test_data", RshiftTensor.test_data)
@XfailIfNoCorstone320
def test_bitwise_right_shift_tensor_u85_BI(test_data):
    pipeline = EthosU85PipelineBI[scalar_input_t](
        RshiftTensor(),
        test_data(),
        RshiftTensor.torch_op,
        RshiftTensor.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
