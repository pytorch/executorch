# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

test_data_suite = {
    "rank2_rand": lambda: (
        torch.randint(-30, 30, (5, 9), dtype=torch.float32),
        torch.randint(0, 9, (9,), dtype=torch.float32),
        0,
        2,
    ),
    "rank2_zeros": lambda: (
        torch.rand((3, 2), dtype=torch.float32),
        torch.randint(0, 4, (2,), dtype=torch.float32),
        0,
        0,
    ),
    "rank3_rand": lambda: (
        torch.rand((2, 4, 5), dtype=torch.float32),
        torch.randint(-5, 5, (2, 5), dtype=torch.float32),
        1,
        0,
    ),
    "rank3_ones": lambda: (
        torch.ones((2, 3, 3), dtype=torch.float32),
        torch.rand((2, 3), dtype=torch.float32),
        2,
        2,
    ),
    "rank4_rand": lambda: (
        torch.rand((1, 2, 4, 5), dtype=torch.float32),
        torch.rand((2, 4, 5), dtype=torch.float32),
        0,
        0,
    ),
    "rank4_ones": lambda: (
        torch.ones((2, 3, 3, 2), dtype=torch.float32),
        torch.randint(-5, 5, (2, 3, 2), dtype=torch.float32),
        2,
        -1,
    ),
    "rank5_ones": lambda: (
        torch.ones((3, 4, 20, 9, 5), dtype=torch.float32),
        torch.randn((3, 4, 20, 9), dtype=torch.float32),
        4,
        1,
    ),
    "rank6_rand": lambda: (
        torch.rand((1, 2, 3, 4, 2, 1), dtype=torch.float32),
        torch.randn((2, 3, 4, 2, 1), dtype=torch.float32),
        0,
        0,
    ),
}
test_data_suite_bf16 = {
    "rank2_rand_bf16": lambda: (
        torch.rand((4, 6), dtype=torch.bfloat16),
        torch.rand((6,), dtype=torch.bfloat16),
        0,
        1,
    ),
    "rank3_ones_bf16": lambda: (
        torch.ones((2, 3, 4), dtype=torch.bfloat16),
        torch.rand((2, 4), dtype=torch.bfloat16),
        1,
        0,
    ),
}


class SelectScatter(torch.nn.Module):
    fp_aten_op = "torch.ops.aten.select_scatter.default"
    int_aten_ops = [
        "torch.ops.aten.arange.start_step",
        "torch.ops.aten.view_copy.default",
        "torch.ops.aten.unsqueeze_copy.default",
        "torch.ops.aten.expand_copy.default",
        "torch.ops.aten.where.self",
        "torch.ops.aten.eq.Tensor",
    ]
    fp_exir_op = ["executorch_exir_dialects_edge__ops_aten_select_scatter_default"]
    int_exir_ops = [
        "executorch_exir_dialects_edge__ops_aten_eq_Tensor",
        "executorch_exir_dialects_edge__ops_aten_where_self",
        "executorch_exir_dialects_edge__ops_aten_arange_start_step",
        "executorch_exir_dialects_edge__ops_aten_view_copy_default",
        "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default",
        "executorch_exir_dialects_edge__ops_aten_expand_copy_default",
    ]
    u55_not_supported = {
        "executorch_exir_dialects_edge__ops_aten_eq_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_where_self": 1,
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor, dim: int, index: int):
        return x.select_scatter(y, dim, index)


input_t = Tuple[torch.Tensor, torch.Tensor, int, int]


@common.parametrize("test_module", test_data_suite | test_data_suite_bf16)
def test_select_scatter_tosa_FP(test_module: input_t):
    pipeline = TosaPipelineFP[input_t](
        SelectScatter(),
        test_module(),
        aten_op=SelectScatter.fp_aten_op,
        exir_op=SelectScatter.fp_exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_module", test_data_suite)
def test_select_scatter_tosa_INT(test_module: input_t):
    pipeline = TosaPipelineINT[input_t](
        SelectScatter(),
        test_module(),
        aten_op=SelectScatter.int_aten_ops,
        exir_op=SelectScatter.int_exir_ops,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_suite)
def test_select_scatter_u55_INT(test_module: input_t):
    # select_scatter is not supported on U55
    pipeline = OpNotSupportedPipeline[input_t](
        SelectScatter(),
        test_module(),
        SelectScatter.u55_not_supported,
        quantize=True,
        u55_subset=True,
        n_expected_delegates=1,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_module", test_data_suite)
def test_select_scatter_u85_INT(test_module: input_t):
    pipeline = EthosU85PipelineINT[input_t](
        SelectScatter(),
        test_module(),
        aten_ops=SelectScatter.int_aten_ops,
        exir_ops=SelectScatter.int_exir_ops,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_suite)
def test_select_scatter_vgf_no_quant(test_module: input_t):
    pipeline = VgfPipeline[input_t](
        SelectScatter(),
        test_module(),
        aten_op=SelectScatter.fp_aten_op,
        exir_op=SelectScatter.fp_exir_op,
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_suite)
def test_select_scatter_vgf_quant(test_module: input_t):
    pipeline = VgfPipeline[input_t](
        SelectScatter(),
        test_module(),
        aten_op=SelectScatter.int_aten_ops,
        exir_op=SelectScatter.int_exir_ops,
        quantize=True,
    )
    pipeline.run()
