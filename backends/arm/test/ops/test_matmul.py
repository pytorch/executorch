# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op_mm = "torch.ops.aten.matmul.default"
exir_op_mm = "executorch_exir_dialects_edge__ops_aten_matmul_default"
input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x


class MatMul(torch.nn.Module):
    test_data_generators = {
        "rand_rand_3d": lambda: (torch.rand(2, 3, 5), torch.rand(2, 5, 2)),
        "rand_rand_4d": lambda: (torch.rand(1, 2, 3, 5), torch.rand(1, 2, 5, 2)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.matmul(x, y)


class MatMulSingleInput(torch.nn.Module):
    test_data_generators = {
        "rand_3d": lambda: (torch.rand(2, 5, 5),),
        "rand_4d": lambda: (torch.rand(1, 2, 5, 5),),
    }

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, x)


class MatMulCombo(torch.nn.Module):
    test_data_generators = {
        "rand_rand_rand_3d": lambda: (
            torch.rand(2, 5, 5),
            torch.rand(2, 5, 2),
            torch.rand(2, 2, 5),
        ),
        "rand_rand_rand_4d": lambda: (
            torch.rand(1, 2, 5, 5),
            torch.rand(1, 2, 5, 2),
            torch.rand(1, 2, 2, 5),
        ),
    }

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        y1 = torch.matmul(x1, x1)
        y2 = torch.matmul(x2, x3)
        return y1 + y2


@common.parametrize("test_data", MatMul.test_data_generators)
def test_matmul_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](MatMul(), test_data(), aten_op_mm, exir_op_mm)
    pipeline.run()


@common.parametrize("test_data", MatMulSingleInput.test_data_generators)
def test_matmul_single_input_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](
        MatMulSingleInput(), test_data(), aten_op_mm, exir_op_mm
    )
    pipeline.run()


@common.parametrize("test_data", MatMulCombo.test_data_generators)
def test_matmul_combo_tosa_MI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](
        MatMulCombo(), test_data(), aten_op_mm, exir_op_mm
    )
    pipeline.run()


@common.parametrize("test_data", MatMul.test_data_generators)
def test_matmul_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](
        MatMul(), test_data(), aten_op_mm, exir_op_mm, qtol=1
    )
    pipeline.run()


@common.parametrize("test_data", MatMulSingleInput.test_data_generators)
def test_matmul_single_input_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](
        MatMulSingleInput(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", MatMulCombo.test_data_generators)
def test_matmul_combo_tosa_BI(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](
        MatMulCombo(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", MatMul.test_data_generators)
@common.XfailIfNoCorstone300
def test_matmul_u55_BI(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        MatMul(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", MatMulSingleInput.test_data_generators)
@common.XfailIfNoCorstone300
def test_matmul_single_input_u55_BI(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        MatMulSingleInput(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", MatMulCombo.test_data_generators)
@common.XfailIfNoCorstone300
def test_matmul_combo_u55_BI(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        MatMulCombo(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", MatMul.test_data_generators)
@common.XfailIfNoCorstone320
def test_matmul_u85_BI(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        MatMul(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", MatMulSingleInput.test_data_generators)
@common.XfailIfNoCorstone320
def test_matmul_single_input_u85_BI(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        MatMulSingleInput(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", MatMulCombo.test_data_generators)
@common.XfailIfNoCorstone320
def test_matmul_combo_u85_BI(test_data: input_t1):
    pipeline = EthosU85PipelineBI[input_t1](
        MatMulCombo(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
