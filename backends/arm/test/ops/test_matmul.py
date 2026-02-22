# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op_mm = "torch.ops.aten.matmul.default"
exir_op_mm = "executorch_exir_dialects_edge__ops_aten_matmul_default"

input_t = Tuple[torch.Tensor, ...]
input_factory_t = Callable[[], input_t]
test_case_t = Callable[[], Tuple[torch.nn.Module, input_factory_t]]


class MatMulDoubleInput(torch.nn.Module):
    test_data = {
        "double_input_randn_rand_1d_1d": lambda: (
            MatMulDoubleInput(),
            lambda: (torch.randn(5), torch.rand(5)),
        ),
        "double_input_randn_rand_2d_2d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                (1 << 30) * torch.randn(5, 5),
                torch.rand(5, 2),
            ),
        ),
        "double_input_randn_rand_2d_1d": lambda: (
            MatMulDoubleInput(),
            lambda: (torch.randn(5, 5), torch.rand(5)),
        ),
        "double_input_randn_rand_1d_2d": lambda: (
            MatMulDoubleInput(),
            lambda: (torch.randn(5), torch.rand(5, 2)),
        ),
        "double_input_randn_rand_3d_3d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.randn(2, 3, 5),
                torch.rand(2, 5, 2),
            ),
        ),
        "double_input_randn_rand_3d_1d": lambda: (
            MatMulDoubleInput(),
            lambda: (torch.randn(2, 3, 5), torch.rand(5)),
        ),
        "double_input_randn_rand_3d_2d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                (1 << 30) * torch.randn(2, 3, 5),
                torch.rand(5, 2),
            ),
        ),
        "double_input_randn_rand_1d_3d": lambda: (
            MatMulDoubleInput(),
            lambda: (torch.randn(5), torch.rand(2, 5, 3)),
        ),
        "double_input_randn_rand_2d_3d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.randn(3, 5),
                torch.rand(2, 5, 3),
            ),
        ),
        "double_input_randn_rand_4d_4d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.randn(1, 2, 3, 5),
                torch.rand(1, 2, 5, 2),
            ),
        ),
        "double_input_randn_rand_4d_1d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.randn(1, 2, 3, 5),
                torch.rand(5),
            ),
        ),
        "double_input_randn_rand_4d_2d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.randn(1, 2, 3, 5),
                torch.rand(5, 3),
            ),
        ),
        "double_input_randn_rand_4d_3d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                (1 << 30) * torch.randn(1, 2, 3, 5),
                torch.rand(2, 5, 3),
            ),
        ),
        "double_input_randn_rand_3d_4d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.randn(4, 3, 5),
                torch.rand(2, 4, 5, 3),
            ),
        ),
        "double_input_randn_rand_2d_4d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.randn(3, 5),
                torch.rand(2, 4, 5, 3),
            ),
        ),
        "double_input_randn_rand_1d_4d": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.randn(5),
                torch.rand(2, 4, 5, 3),
            ),
        ),
    }

    test_data_fp16 = {
        "double_input_rand_rand_2d_fp16": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.rand(4, 4, dtype=torch.float16),
                torch.rand(4, 3, dtype=torch.float16),
            ),
        ),
    }

    test_data_bf16 = {
        "double_input_rand_rand_2d_bf16": lambda: (
            MatMulDoubleInput(),
            lambda: (
                torch.rand(4, 4, dtype=torch.bfloat16),
                torch.rand(4, 3, dtype=torch.bfloat16),
            ),
        ),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.matmul(x, y)


class MatMulSingleInput(torch.nn.Module):
    test_data = {
        "single_input_randn_1d": lambda: (
            MatMulSingleInput(),
            lambda: (torch.randn(5),),
        ),
        "single_input_randn_2d": lambda: (
            MatMulSingleInput(),
            lambda: (torch.randn(5, 5),),
        ),
        "single_input_randn_3d": lambda: (
            MatMulSingleInput(),
            lambda: (torch.randn(2, 5, 5),),
        ),
        "single_input_randn_4d": lambda: (
            MatMulSingleInput(),
            lambda: (torch.randn(1, 2, 5, 5),),
        ),
    }

    test_data_fp16 = {
        "single_input_rand_2d_fp16": lambda: (
            MatMulSingleInput(),
            lambda: (torch.rand(4, 4, dtype=torch.float16),),
        ),
    }

    test_data_bf16 = {
        "single_input_rand_2d_bf16": lambda: (
            MatMulSingleInput(),
            lambda: (torch.rand(4, 4, dtype=torch.bfloat16),),
        ),
    }

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, x)


class MatMulCombo(torch.nn.Module):
    test_data = {
        "combo_rand_randn_rand_2d": lambda: (
            MatMulCombo(),
            lambda: (
                torch.rand(5, 5),
                10e8 * torch.randn(5, 2),
                torch.rand(2, 5),
            ),
        ),
        "combo_rand_randn_rand_3d": lambda: (
            MatMulCombo(),
            lambda: (
                torch.rand(2, 5, 5),
                10e12 * torch.randn(2, 5, 2),
                torch.rand(2, 2, 5),
            ),
        ),
        "combo_rand_randn_rand_4d": lambda: (
            MatMulCombo(),
            lambda: (
                torch.rand(1, 2, 5, 5),
                torch.randn(1, 2, 5, 2),
                torch.rand(1, 2, 2, 5),
            ),
        ),
    }

    test_data_fp16 = {
        "combo_rand_rand_rand_2d_fp16": lambda: (
            MatMulCombo(),
            lambda: (
                torch.rand(4, 4, dtype=torch.float16),
                torch.rand(4, 3, dtype=torch.float16),
                torch.rand(3, 4, dtype=torch.float16),
            ),
        ),
    }

    test_data_bf16 = {
        "combo_rand_rand_rand_2d_bf16": lambda: (
            MatMulCombo(),
            lambda: (
                torch.rand(4, 4, dtype=torch.bfloat16),
                torch.rand(4, 3, dtype=torch.bfloat16),
                torch.rand(3, 4, dtype=torch.bfloat16),
            ),
        ),
    }

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        y1 = torch.matmul(x1, x2)
        return torch.matmul(y1, x3)


test_suite = (
    MatMulDoubleInput.test_data | MatMulSingleInput.test_data | MatMulCombo.test_data
)
test_suite_fp16 = (
    MatMulDoubleInput.test_data_fp16
    | MatMulSingleInput.test_data_fp16
    | MatMulCombo.test_data_fp16
)
test_suite_bf16 = (
    MatMulDoubleInput.test_data_bf16
    | MatMulSingleInput.test_data_bf16
    | MatMulCombo.test_data_bf16
)
xfails = {
    "double_input_randn_rand_1d_1d": "aten.dot.default is not supported",
    "double_input_randn_rand_2d_1d": "aten.mv.default is not supported",
    "double_input_randn_rand_3d_1d": "aten.mv.default is not supported",
    "double_input_randn_rand_4d_1d": "aten.mv.default is not supported",
    "single_input_randn_1d": "aten.dot.default is not supported",
}


@common.parametrize("test_case", test_suite | test_suite_fp16 | test_suite_bf16)
def test_matmul_tosa_FP(test_case: test_case_t):
    model, inputs = test_case()
    pipeline = TosaPipelineFP[input_t](
        model, inputs(), aten_op_mm, exir_op_mm, tosa_extensions=["bf16"]
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
def test_matmul_tosa_INT(test_case: test_case_t):
    model, inputs = test_case()
    pipeline = TosaPipelineINT[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.XfailIfNoCorstone300
def test_matmul_u55_INT(test_case: test_case_t):
    model, inputs = test_case()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.XfailIfNoCorstone320
def test_matmul_u85_INT(test_case: test_case_t):
    model, inputs = test_case()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite | test_suite_fp16)
@common.SkipIfNoModelConverter
def test_matmul_vgf_no_quant(test_case: test_case_t):
    model, inputs = test_case()
    pipeline = VgfPipeline[input_t](
        model,
        inputs(),
        aten_op_mm,
        exir_op_mm,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.SkipIfNoModelConverter
def test_matmul_vgf_quant(test_case: test_case_t):
    model, inputs = test_case()
    pipeline = VgfPipeline[input_t](
        model, inputs(), [], exir_op_mm, quantize=True, run_on_vulkan_runtime=False
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
def test_matmul_tosa_INT_a16w8(test_case: test_case_t):
    """Test matmul with 16A8W quantization for TOSA INT."""
    model, inputs = test_case()
    pipeline = TosaPipelineINT[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@pytest.mark.xfail(
    reason="Vela compilation fails with 'Non-passthrough operation' for int16 matmul operations"
)
@common.XfailIfNoCorstone300
def test_matmul_u55_INT_a16w8(test_case: test_case_t):
    """Test matmul with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    model, inputs = test_case()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
        a16w8_quantization=True,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.XfailIfNoCorstone320
def test_matmul_u85_INT_a16w8(test_case: test_case_t):
    """Test matmul with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    model, inputs = test_case()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
        a16w8_quantization=True,
    )
    pipeline.run()
