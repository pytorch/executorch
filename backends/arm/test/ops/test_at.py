# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op_mm = "torch.ops.aten.matmul.default"
exir_op_mm = "executorch_exir_dialects_edge__ops_aten_matmul_default"
input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x


class AtMatMulSingleInput(torch.nn.Module):
    test_data_generators = {
        "rand_3d": lambda: (torch.rand(2, 5, 5),),
        "rand_4d": lambda: (torch.rand(1, 2, 5, 5),),
    }

    def forward(self, x: torch.Tensor):
        return x @ x


class AtMatMulDoubleInput(torch.nn.Module):
    test_data_generators = {
        "rand_rand_3d": lambda: (torch.rand(2, 3, 5), torch.rand(2, 5, 2)),
        "rand_rand_4d": lambda: (torch.rand(1, 2, 3, 5), torch.rand(1, 2, 5, 2)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x @ y


class AtMatMulMixedPattern1(torch.nn.Module):
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


class AtMatMulMixedPattern2(torch.nn.Module):
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
        return y1 @ y2


@common.parametrize("test_data", AtMatMulSingleInput.test_data_generators)
def test_matmul_tosa_FP_at_single_input(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        AtMatMulSingleInput(), test_data(), aten_op_mm, exir_op_mm
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulDoubleInput.test_data_generators)
def test_matmul_tosa_FP_at_double_input(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        AtMatMulDoubleInput(), test_data(), aten_op_mm, exir_op_mm
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulMixedPattern1.test_data_generators)
def test_matmul_tosa_FP_at_mixed_pattern1(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        AtMatMulMixedPattern1(), test_data(), aten_op_mm, exir_op_mm
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulMixedPattern2.test_data_generators)
def test_matmul_tosa_FP_at_mixed_pattern2(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        AtMatMulMixedPattern2(), test_data(), aten_op_mm, exir_op_mm
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulSingleInput.test_data_generators)
def test_matmul_tosa_INT_at_single_input(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        AtMatMulSingleInput(), test_data(), aten_op_mm, exir_op_mm
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulDoubleInput.test_data_generators)
def test_matmul_tosa_INT_at_double_input(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        AtMatMulDoubleInput(), test_data(), aten_op_mm, exir_op_mm
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulMixedPattern1.test_data_generators)
def test_matmul_tosa_INT_at_mixed_pattern1(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        AtMatMulMixedPattern1(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulMixedPattern2.test_data_generators)
def test_matmul_tosa_INT_at_mixed_pattern2(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        AtMatMulMixedPattern2(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulSingleInput.test_data_generators)
@common.SkipIfNoModelConverter
def test_matmul_vgf_no_quant_at_single_input(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AtMatMulSingleInput(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulDoubleInput.test_data_generators)
@common.SkipIfNoModelConverter
def test_matmul_vgf_no_quant_at_double_input(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AtMatMulDoubleInput(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulMixedPattern1.test_data_generators)
@common.SkipIfNoModelConverter
def test_matmul_vgf_no_quant_at_mixed_pattern1(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AtMatMulMixedPattern1(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulMixedPattern2.test_data_generators)
@common.SkipIfNoModelConverter
def test_matmul_vgf_no_quant_at_mixed_pattern2(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AtMatMulMixedPattern2(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulSingleInput.test_data_generators)
@common.SkipIfNoModelConverter
def test_matmul_vgf_quant_at_single_input(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AtMatMulSingleInput(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulDoubleInput.test_data_generators)
@common.SkipIfNoModelConverter
def test_matmul_vgf_quant_at_double_input(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AtMatMulDoubleInput(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulMixedPattern1.test_data_generators)
@common.SkipIfNoModelConverter
def test_matmul_vgf_quant_at_mixed_pattern1(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AtMatMulMixedPattern1(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        qtol=1,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", AtMatMulMixedPattern2.test_data_generators)
@common.SkipIfNoModelConverter
def test_matmul_vgf_quant_at_mixed_pattern2(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AtMatMulMixedPattern2(),
        test_data(),
        aten_op_mm,
        exir_op_mm,
        qtol=1,
        quantize=True,
    )
    pipeline.run()
