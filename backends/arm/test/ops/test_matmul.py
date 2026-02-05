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
test_case_t = Tuple[torch.nn.Module, input_factory_t]


class MatMulDoubleInput(torch.nn.Module):
    test_data = {
        "randn_rand_1d_1d": lambda: (torch.randn(5), torch.rand(5)),
        "randn_rand_2d_2d": lambda: ((1 << 30) * torch.randn(5, 5), torch.rand(5, 2)),
        "randn_rand_2d_1d": lambda: (torch.randn(5, 5), torch.rand(5)),
        "randn_rand_1d_2d": lambda: (torch.randn(5), torch.rand(5, 2)),
        "randn_rand_3d_3d": lambda: (torch.randn(2, 3, 5), torch.rand(2, 5, 2)),
        "randn_rand_3d_1d": lambda: (torch.randn(2, 3, 5), torch.rand(5)),
        "randn_rand_3d_2d": lambda: (
            (1 << 30) * torch.randn(2, 3, 5),
            torch.rand(5, 2),
        ),
        "randn_rand_1d_3d": lambda: (torch.randn(5), torch.rand(2, 5, 3)),
        "randn_rand_2d_3d": lambda: (torch.randn(3, 5), torch.rand(2, 5, 3)),
        "randn_rand_4d_4d": lambda: (torch.randn(1, 2, 3, 5), torch.rand(1, 2, 5, 2)),
        "randn_rand_4d_1d": lambda: (torch.randn(1, 2, 3, 5), torch.rand(5)),
        "randn_rand_4d_2d": lambda: (torch.randn(1, 2, 3, 5), torch.rand(5, 3)),
        "randn_rand_4d_3d": lambda: (
            (1 << 30) * torch.randn(1, 2, 3, 5),
            torch.rand(2, 5, 3),
        ),
        "randn_rand_3d_4d": lambda: (torch.randn(4, 3, 5), torch.rand(2, 4, 5, 3)),
        "randn_rand_2d_4d": lambda: (torch.randn(3, 5), torch.rand(2, 4, 5, 3)),
        "randn_rand_1d_4d": lambda: (
            torch.randn(
                5,
            ),
            torch.rand(2, 4, 5, 3),
        ),
    }

    test_data_bf16 = {
        "rand_rand_2d_bf16": lambda: (
            torch.rand(4, 4, dtype=torch.bfloat16),
            torch.rand(4, 3, dtype=torch.bfloat16),
        ),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.matmul(x, y)


class MatMulSingleInput(torch.nn.Module):
    test_data = {
        "randn_1d": lambda: (torch.randn(5),),
        "randn_2d": lambda: (torch.randn(5, 5),),
        "randn_3d": lambda: (torch.randn(2, 5, 5),),
        "randn_4d": lambda: (torch.randn(1, 2, 5, 5),),
    }

    test_data_bf16 = {
        "rand_2d_bf16": lambda: (torch.rand(4, 4, dtype=torch.bfloat16),),
    }

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, x)


class MatMulCombo(torch.nn.Module):
    test_data = {
        "rand_randn_rand_2d": lambda: (
            torch.rand(5, 5),
            10e8 * torch.randn(5, 2),
            torch.rand(2, 5),
        ),
        "rand_randn_rand_3d": lambda: (
            torch.rand(2, 5, 5),
            10e12 * torch.randn(2, 5, 2),
            torch.rand(2, 2, 5),
        ),
        "rand_randn_rand_4d": lambda: (
            torch.rand(1, 2, 5, 5),
            torch.randn(1, 2, 5, 2),
            torch.rand(1, 2, 2, 5),
        ),
    }

    test_data_bf16 = {
        "rand_rand_rand_2d_bf16": lambda: (
            torch.rand(4, 4, dtype=torch.bfloat16),
            torch.rand(4, 3, dtype=torch.bfloat16),
            torch.rand(3, 4, dtype=torch.bfloat16),
        ),
    }

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        y1 = torch.matmul(x1, x2)
        return torch.matmul(y1, x3)


test_suite_double_input = {
    f"double_input_{name}": (MatMulDoubleInput(), inputs)
    for name, inputs in MatMulDoubleInput.test_data.items()
}
test_suite_bf16_double_input = {
    f"double_input_bf16_{name}": (MatMulDoubleInput(), inputs)
    for name, inputs in MatMulDoubleInput.test_data_bf16.items()
}
test_suite_single_input = {
    f"single_input_{name}": (MatMulSingleInput(), inputs)
    for name, inputs in MatMulSingleInput.test_data.items()
}
test_suite_bf16_single_input = {
    f"single_input_bf16_{name}": (MatMulSingleInput(), inputs)
    for name, inputs in MatMulSingleInput.test_data_bf16.items()
}
test_suite_combo = {
    f"combo_{name}": (MatMulCombo(), inputs)
    for name, inputs in MatMulCombo.test_data.items()
}
test_suite_bf16_combo = {
    f"combo_bf16_{name}": (MatMulCombo(), inputs)
    for name, inputs in MatMulCombo.test_data_bf16.items()
}
test_suite = test_suite_double_input | test_suite_single_input | test_suite_combo
test_suite_bf16 = (
    test_suite_bf16_double_input | test_suite_bf16_single_input | test_suite_bf16_combo
)
xfails = {
    "double_input_randn_rand_1d_1d": "aten.dot.default is not supported",
    "double_input_randn_rand_2d_1d": "aten.mv.default is not supported",
    "double_input_randn_rand_3d_1d": "aten.mv.default is not supported",
    "double_input_randn_rand_4d_1d": "aten.mv.default is not supported",
    "single_input_randn_1d": "aten.dot.default is not supported",
}


@common.parametrize("test_case", test_suite | test_suite_bf16)
def test_matmul_tosa_FP(test_case: test_case_t):
    model, inputs = test_case
    pipeline = TosaPipelineFP[input_t](
        model, inputs(), aten_op_mm, exir_op_mm, tosa_extensions=["bf16"]
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
def test_matmul_tosa_INT(test_case: test_case_t):
    model, inputs = test_case
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
    model, inputs = test_case
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
    model, inputs = test_case
    pipeline = EthosU85PipelineINT[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite)
@common.SkipIfNoModelConverter
def test_matmul_vgf_no_quant(test_case: test_case_t):
    model, inputs = test_case
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
    model, inputs = test_case
    pipeline = VgfPipeline[input_t](
        model, inputs(), [], exir_op_mm, quantize=True, run_on_vulkan_runtime=False
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
def test_matmul_tosa_INT_a16w8(test_case: test_case_t):
    """Test matmul with 16A8W quantization for TOSA INT."""
    model, inputs = test_case
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
    model, inputs = test_case
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
    model, inputs = test_case
    pipeline = EthosU85PipelineINT[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
        a16w8_quantization=True,
    )
    pipeline.run()
