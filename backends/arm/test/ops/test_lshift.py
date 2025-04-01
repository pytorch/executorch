# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
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
from parameterized import parameterized

scalar_input_t = tuple[torch.Tensor, int]


class LshiftScalar(torch.nn.Module):
    torch_op_MI = "torch.ops.aten.__lshift__.Scalar"
    torch_op_BI = "torch.ops.aten.bitwise_left_shift.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_left_shift_Tensor"
    test_data = [
        ((torch.randint(-8, 8, (1, 12, 3, 4), dtype=torch.int8), 1),),
        ((torch.randint(-100, 100, (1, 5, 3, 4), dtype=torch.int16), 5),),
        ((torch.randint(-100, 100, (1, 5, 3, 4), dtype=torch.int32), 2),),
    ]

    def forward(self, x: torch.Tensor, shift: int):
        return x << shift


tensor_input_t = tuple[torch.Tensor, torch.Tensor]


class LshiftTensor(torch.nn.Module):
    torch_op = "torch.ops.aten.bitwise_left_shift.Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_left_shift_Tensor"
    test_data = [
        (
            (
                torch.randint(-8, 8, (3, 3), dtype=torch.int8),
                torch.randint(0, 4, (3, 3), dtype=torch.int8),
            ),
        ),
        (
            (
                torch.randint(-1024, 1024, (3, 3, 3), dtype=torch.int16),
                torch.randint(0, 5, (3, 3, 3), dtype=torch.int16),
            ),
        ),
        (
            (
                torch.randint(0, 127, (1, 2, 3, 3), dtype=torch.int32),
                torch.randint(0, 5, (1, 2, 3, 3), dtype=torch.int32),
            ),
        ),
    ]

    def forward(self, x: torch.Tensor, shift: torch.Tensor):
        return x.bitwise_left_shift(shift)


@parameterized.expand(LshiftScalar.test_data)
def test_lshift_scalar_tosa_MI(test_data):
    TosaPipelineMI[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_MI,
        LshiftScalar.exir_op,
    ).run()


@parameterized.expand(LshiftScalar.test_data)
def test_lshift_scalar_tosa_BI(test_data):
    pipeline = TosaPipelineBI[scalar_input_t](
        LshiftScalar(), test_data, LshiftScalar.torch_op_BI, LshiftScalar.exir_op
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@parameterized.expand(LshiftScalar.test_data)
@XfailIfNoCorstone300
def test_lshift_scalar_tosa_u55(test_data):
    pipeline = EthosU55PipelineBI[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_BI,
        LshiftScalar.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@parameterized.expand(LshiftScalar.test_data)
@XfailIfNoCorstone320
def test_lshift_scalar_tosa_u85(test_data):
    pipeline = EthosU85PipelineBI[scalar_input_t](
        LshiftScalar(),
        test_data,
        LshiftScalar.torch_op_BI,
        LshiftScalar.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@parameterized.expand(LshiftTensor.test_data)
def test_lshift_tensor_tosa_MI(test_data):
    TosaPipelineMI[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
    ).run()


@parameterized.expand(LshiftTensor.test_data)
def test_lshift_tensor_tosa_BI(test_data):
    pipeline = TosaPipelineBI[scalar_input_t](
        LshiftTensor(), test_data, LshiftTensor.torch_op, LshiftTensor.exir_op
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@parameterized.expand(LshiftTensor.test_data)
@XfailIfNoCorstone300
def test_lshift_tensor_tosa_u55(test_data):
    pipeline = EthosU55PipelineBI[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@parameterized.expand(LshiftTensor.test_data)
@XfailIfNoCorstone320
def test_lshift_tensor_tosa_u85(test_data):
    pipeline = EthosU85PipelineBI[scalar_input_t](
        LshiftTensor(),
        test_data,
        LshiftTensor.torch_op,
        LshiftTensor.exir_op,
        run_on_fvp=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
