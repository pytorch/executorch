# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from executorch.backends.arm.quantizer import get_symmetric_a16w8_quantization_config
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op_mm = "torch.ops.aten.matmul.default"
exir_op_mm_2d = "executorch_exir_dialects_edge__ops_aten_mm_default"
exir_op_bmm = "executorch_exir_dialects_edge__ops_aten_bmm_default"
exir_op_mul = "executorch_exir_dialects_edge__ops_aten_mul_Tensor"
exir_op_sum = "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList"
exir_op_view = "executorch_exir_dialects_edge__ops_aten_view_copy_default"

input_t = Tuple[torch.Tensor, ...]
input_factory_t = Callable[[], input_t]

EXIR_OPS_MM = (exir_op_mm_2d,)
EXIR_OPS_BMM = (exir_op_bmm,)
EXIR_OPS_MUL_SUM = (exir_op_mul, exir_op_sum)
EXIR_OPS_MUL_SUM_VIEW = (exir_op_mul, exir_op_sum, exir_op_view)


@dataclass(frozen=True)
class MatMulTestCase:
    module: torch.nn.Module
    input_factory: input_factory_t
    exir_ops: tuple[str, ...]
    u55_a16w8_non_delegated_ops: dict[str, int]
    u55_a16w8_n_delegates: int = 0


def _make_test_case(
    module: torch.nn.Module,
    input_factory: input_factory_t,
    exir_ops: tuple[str, ...],
    u55_a16w8_non_delegated_ops: dict[str, int] | None = None,
    u55_a16w8_n_delegates: int = 0,
) -> MatMulTestCase:
    non_delegated_ops = u55_a16w8_non_delegated_ops
    if non_delegated_ops is None:
        non_delegated_ops = {op: exir_ops.count(op) for op in dict.fromkeys(exir_ops)}

    return MatMulTestCase(
        module=module,
        input_factory=input_factory,
        exir_ops=exir_ops,
        u55_a16w8_non_delegated_ops=dict(non_delegated_ops),
        u55_a16w8_n_delegates=u55_a16w8_n_delegates,
    )


test_case_t = Callable[[], MatMulTestCase]


class MatMulDoubleInput(torch.nn.Module):
    test_data = {
        "double_input_randn_rand_1d_1d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (torch.randn(5), torch.rand(5)),
            EXIR_OPS_MUL_SUM,
        ),
        "double_input_randn_rand_2d_2d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                (1 << 30) * torch.randn(5, 5),
                torch.rand(5, 2),
            ),
            EXIR_OPS_MM,
        ),
        "double_input_randn_rand_2d_1d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (torch.randn(5, 5), torch.rand(5)),
            EXIR_OPS_MUL_SUM,
        ),
        "double_input_randn_rand_1d_2d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (torch.randn(5), torch.rand(5, 2)),
            EXIR_OPS_MM,
            u55_a16w8_n_delegates=2,
        ),
        "double_input_randn_rand_3d_3d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.randn(2, 3, 5),
                torch.rand(2, 5, 2),
            ),
            EXIR_OPS_BMM,
        ),
        "double_input_randn_rand_3d_1d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (torch.randn(2, 3, 5), torch.rand(5)),
            EXIR_OPS_MUL_SUM_VIEW,
            u55_a16w8_non_delegated_ops={
                exir_op_mul: 1,
                exir_op_sum: 1,
                exir_op_view: 2,
            },
        ),
        "double_input_randn_rand_3d_2d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                (1 << 30) * torch.randn(2, 3, 5),
                torch.rand(5, 2),
            ),
            EXIR_OPS_MM,
        ),
        "double_input_randn_rand_1d_3d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (torch.randn(5), torch.rand(2, 5, 3)),
            EXIR_OPS_BMM,
            u55_a16w8_n_delegates=1,
        ),
        "double_input_randn_rand_2d_3d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.randn(3, 5),
                torch.rand(2, 5, 3),
            ),
            EXIR_OPS_BMM,
            u55_a16w8_n_delegates=1,
        ),
        "double_input_randn_rand_4d_4d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.randn(1, 2, 3, 5),
                torch.rand(1, 2, 5, 2),
            ),
            EXIR_OPS_BMM,
        ),
        "double_input_randn_rand_4d_1d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.randn(1, 2, 3, 5),
                torch.rand(5),
            ),
            EXIR_OPS_MUL_SUM_VIEW,
            u55_a16w8_non_delegated_ops={
                exir_op_mul: 1,
                exir_op_sum: 1,
                exir_op_view: 2,
            },
        ),
        "double_input_randn_rand_4d_2d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.randn(1, 2, 3, 5),
                torch.rand(5, 3),
            ),
            EXIR_OPS_MM,
        ),
        "double_input_randn_rand_4d_3d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                (1 << 30) * torch.randn(1, 2, 3, 5),
                torch.rand(2, 5, 3),
            ),
            EXIR_OPS_BMM,
            u55_a16w8_n_delegates=1,
        ),
        "double_input_randn_rand_3d_4d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.randn(4, 3, 5),
                torch.rand(2, 4, 5, 3),
            ),
            EXIR_OPS_BMM,
            u55_a16w8_n_delegates=1,
        ),
        "double_input_randn_rand_2d_4d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.randn(3, 5),
                torch.rand(2, 4, 5, 3),
            ),
            EXIR_OPS_BMM,
            u55_a16w8_n_delegates=1,
        ),
        "double_input_randn_rand_1d_4d": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.randn(5),
                torch.rand(2, 4, 5, 3),
            ),
            EXIR_OPS_BMM,
            u55_a16w8_n_delegates=1,
        ),
    }

    test_data_fp16 = {
        "double_input_rand_rand_2d_fp16": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.rand(4, 4, dtype=torch.float16),
                torch.rand(4, 3, dtype=torch.float16),
            ),
            EXIR_OPS_MM,
        ),
    }

    test_data_bf16 = {
        "double_input_rand_rand_2d_bf16": lambda: _make_test_case(
            MatMulDoubleInput(),
            lambda: (
                torch.rand(4, 4, dtype=torch.bfloat16),
                torch.rand(4, 3, dtype=torch.bfloat16),
            ),
            EXIR_OPS_MM,
        ),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.matmul(x, y)


class MatMulSingleInput(torch.nn.Module):
    test_data = {
        "single_input_randn_1d": lambda: _make_test_case(
            MatMulSingleInput(),
            lambda: (torch.randn(5),),
            EXIR_OPS_MUL_SUM,
        ),
        "single_input_randn_2d": lambda: _make_test_case(
            MatMulSingleInput(),
            lambda: (torch.randn(5, 5),),
            EXIR_OPS_MM,
        ),
        "single_input_randn_3d": lambda: _make_test_case(
            MatMulSingleInput(),
            lambda: (torch.randn(2, 5, 5),),
            EXIR_OPS_BMM,
        ),
        "single_input_randn_4d": lambda: _make_test_case(
            MatMulSingleInput(),
            lambda: (torch.randn(1, 2, 5, 5),),
            EXIR_OPS_BMM,
        ),
    }

    test_data_fp16 = {
        "single_input_rand_2d_fp16": lambda: _make_test_case(
            MatMulSingleInput(),
            lambda: (torch.rand(4, 4, dtype=torch.float16),),
            EXIR_OPS_MM,
        ),
    }

    test_data_bf16 = {
        "single_input_rand_2d_bf16": lambda: _make_test_case(
            MatMulSingleInput(),
            lambda: (torch.rand(4, 4, dtype=torch.bfloat16),),
            EXIR_OPS_MM,
        ),
    }

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, x)


class MatMulCombo(torch.nn.Module):
    test_data = {
        "combo_rand_randn_rand_2d": lambda: _make_test_case(
            MatMulCombo(),
            lambda: (
                torch.rand(5, 5),
                10e8 * torch.randn(5, 2),
                torch.rand(2, 5),
            ),
            (exir_op_mm_2d, exir_op_mm_2d),
        ),
        "combo_rand_randn_rand_3d": lambda: _make_test_case(
            MatMulCombo(),
            lambda: (
                torch.rand(2, 5, 5),
                10e12 * torch.randn(2, 5, 2),
                torch.rand(2, 2, 5),
            ),
            (exir_op_bmm, exir_op_bmm),
        ),
        "combo_rand_randn_rand_4d": lambda: _make_test_case(
            MatMulCombo(),
            lambda: (
                torch.rand(1, 2, 5, 5),
                torch.randn(1, 2, 5, 2),
                torch.rand(1, 2, 2, 5),
            ),
            (exir_op_bmm, exir_op_bmm),
        ),
    }

    test_data_fp16 = {
        "combo_rand_rand_rand_2d_fp16": lambda: _make_test_case(
            MatMulCombo(),
            lambda: (
                torch.rand(4, 4, dtype=torch.float16),
                torch.rand(4, 3, dtype=torch.float16),
                torch.rand(3, 4, dtype=torch.float16),
            ),
            EXIR_OPS_MM,
        ),
    }

    test_data_bf16 = {
        "combo_rand_rand_rand_2d_bf16": lambda: _make_test_case(
            MatMulCombo(),
            lambda: (
                torch.rand(4, 4, dtype=torch.bfloat16),
                torch.rand(4, 3, dtype=torch.bfloat16),
                torch.rand(3, 4, dtype=torch.bfloat16),
            ),
            EXIR_OPS_MM,
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
    test_data = test_case()
    pipeline = TosaPipelineFP[input_t](
        test_data.module,
        test_data.input_factory(),
        aten_op_mm,
        list(test_data.exir_ops),
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
def test_matmul_tosa_INT(test_case: test_case_t):
    test_data = test_case()
    pipeline = TosaPipelineINT[input_t](
        test_data.module,
        test_data.input_factory(),
        [],
        list(test_data.exir_ops),
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.XfailIfNoCorstone300
def test_matmul_u55_INT(test_case: test_case_t):
    test_data = test_case()
    pipeline = EthosU55PipelineINT[input_t](
        test_data.module,
        test_data.input_factory(),
        [],
        list(test_data.exir_ops),
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.XfailIfNoCorstone320
def test_matmul_u85_INT(test_case: test_case_t):
    test_data = test_case()
    pipeline = EthosU85PipelineINT[input_t](
        test_data.module,
        test_data.input_factory(),
        [],
        list(test_data.exir_ops),
    )
    pipeline.run()


@common.parametrize("test_case", test_suite | test_suite_fp16)
@common.SkipIfNoModelConverter
def test_matmul_vgf_no_quant(test_case: test_case_t):
    test_data = test_case()
    pipeline = VgfPipeline[input_t](
        test_data.module,
        test_data.input_factory(),
        aten_op_mm,
        list(test_data.exir_ops),
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.SkipIfNoModelConverter
def test_matmul_vgf_quant(test_case: test_case_t):
    test_data = test_case()
    pipeline = VgfPipeline[input_t](
        test_data.module,
        test_data.input_factory(),
        [],
        list(test_data.exir_ops),
        quantize=True,
        run_on_vulkan_runtime=False,
    )
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.SkipIfNoModelConverter
def test_matmul_vgf_quant_a16w8(test_case: test_case_t):
    test_data = test_case()
    pipeline = VgfPipeline[input_t](
        test_data.module,
        test_data.input_factory(),
        [],
        list(test_data.exir_ops),
        quantize=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(get_symmetric_a16w8_quantization_config())
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
def test_matmul_tosa_INT_a16w8(test_case: test_case_t):
    """Test matmul with 16A8W quantization for TOSA INT."""
    test_data = test_case()
    pipeline = TosaPipelineINT[input_t](
        test_data.module,
        test_data.input_factory(),
        [],
        list(test_data.exir_ops),
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize("test_case", test_suite)
@common.XfailIfNoCorstone300
def test_matmul_u55_INT_a16w8(test_case: test_case_t):
    """Test matmul with 16A8W quantization on U55 (16-bit activations, 8-bit
    weights).

    U55 does not support matmuls with INT16 inputs, so all matmuls should be
    rejected by the partitioner.

    """
    test_data = test_case()
    test_inputs = test_data.input_factory()

    pipeline = OpNotSupportedPipeline[input_t](
        test_data.module,
        test_inputs,
        non_delegated_ops=test_data.u55_a16w8_non_delegated_ops,
        n_expected_delegates=test_data.u55_a16w8_n_delegates,
        u55_subset=True,
        quantize=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(get_symmetric_a16w8_quantization_config())
    pipeline.run()


@common.parametrize("test_case", test_suite, xfails=xfails)
@common.XfailIfNoCorstone320
def test_matmul_u85_INT_a16w8(test_case: test_case_t):
    """Test matmul with 16A8W quantization on U85 (16-bit activations, 8-bit
    weights)
    """
    test_data = test_case()
    pipeline = EthosU85PipelineINT[input_t](
        test_data.module,
        test_data.input_factory(),
        [],
        list(test_data.exir_ops),
        a16w8_quantization=True,
    )
    pipeline.run()
