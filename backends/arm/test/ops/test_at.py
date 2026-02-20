# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op_mm = "torch.ops.aten.matmul.default"
exir_op_mm = "executorch_exir_dialects_edge__ops_aten_matmul_default"
input_t = Tuple[torch.Tensor, ...]
input_factory_t = Callable[[], input_t]
test_case_t = Tuple[torch.nn.Module, input_factory_t]


class AtMatMulSingleInput(torch.nn.Module):
    test_data = {
        "rand_3d": lambda: (torch.rand(2, 5, 5),),
        "rand_4d": lambda: (torch.rand(1, 2, 5, 5),),
    }

    def forward(self, x: torch.Tensor):
        return x @ x


class AtMatMulDoubleInput(torch.nn.Module):
    test_data = {
        "rand_rand_3d": lambda: (torch.rand(2, 3, 5), torch.rand(2, 5, 2)),
        "rand_rand_4d": lambda: (torch.rand(1, 2, 3, 5), torch.rand(1, 2, 5, 2)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x @ y


class AtMatMulMixedPattern2(torch.nn.Module):
    test_data = {
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


test_suite_single_input = {
    f"single_input_{name}": (AtMatMulSingleInput(), inputs)
    for name, inputs in AtMatMulSingleInput.test_data.items()
}
test_suite_double_input = {
    f"double_input_{name}": (AtMatMulDoubleInput(), inputs)
    for name, inputs in AtMatMulDoubleInput.test_data.items()
}
test_suite_mixed = {
    f"mixed_pattern_{name}": (AtMatMulMixedPattern2(), inputs)
    for name, inputs in AtMatMulMixedPattern2.test_data.items()
}
test_suite = test_suite_single_input | test_suite_double_input | test_suite_mixed


@common.parametrize("test_case", test_suite)
def test_matmul_tosa_FP(test_case: test_case_t):
    model, inputs = test_case
    pipeline = TosaPipelineFP[input_t](model, inputs(), aten_op_mm, exir_op_mm)
    pipeline.run()


@common.parametrize("test_case", test_suite)
def test_matmul_tosa_INT(test_case: test_case_t):
    model, inputs = test_case
    pipeline = TosaPipelineINT[input_t](model, inputs(), [], exir_op_mm)
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


@common.parametrize("test_case", test_suite)
@common.SkipIfNoModelConverter
def test_matmul_vgf_quant(test_case: test_case_t):
    model, inputs = test_case
    pipeline = VgfPipeline[input_t](
        model,
        inputs(),
        [],
        exir_op_mm,
        quantize=True,
    )
    pipeline.run()
