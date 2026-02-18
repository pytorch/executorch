# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

test_data_fp_step1 = {
    "rank2_step1": lambda: (
        torch.rand((5, 9), dtype=torch.float32),
        torch.rand((5, 5), dtype=torch.float32),
        1,
        2,
        7,
        1,
    ),
    "rank4_negative": lambda: (
        torch.rand((1, 2, 4, 5), dtype=torch.float32),
        torch.rand((1, 2, 2, 5), dtype=torch.float32),
        2,
        -3,
        -1,
        1,
    ),
}

test_data_fp_stepN = {
    "rank3_step2": lambda: (
        torch.rand((2, 4, 6), dtype=torch.float32),
        torch.rand((2, 4, 2), dtype=torch.float32),
        2,
        1,
        5,
        2,
    ),
    "rank3_end_none": lambda: (
        torch.rand((3, 5, 4), dtype=torch.float32),
        torch.rand((3, 2, 4), dtype=torch.float32),
        1,
        1,
        None,
        2,
    ),
}

test_data_int_step1 = {
    "rank2_step1_int8": lambda: (
        torch.randint(-5, 5, (5, 9), dtype=torch.int8),
        torch.randint(-5, 5, (5, 5), dtype=torch.int8),
        1,
        2,
        7,
        1,
    ),
}

test_data_int_stepN = {
    "rank3_step2_int32": lambda: (
        torch.randint(-50, 50, (2, 4, 6), dtype=torch.int32),
        torch.randint(-50, 50, (2, 4, 2), dtype=torch.int32),
        2,
        1,
        5,
        2,
    ),
}

test_data_bf16 = {
    "rank2_step1_bf16": lambda: (
        torch.rand((4, 8), dtype=torch.bfloat16),
        torch.rand((4, 3), dtype=torch.bfloat16),
        1,
        2,
        5,
        1,
    ),
}


class SliceScatter(torch.nn.Module):
    fp_aten_op = "torch.ops.aten.slice_scatter.default"
    fp_exir_op = ["executorch_exir_dialects_edge__ops_aten_slice_scatter_default"]
    int_aten_ops_step1 = [
        "torch.ops.aten.slice_copy.Tensor",
        "torch.ops.aten.cat.default",
    ]
    int_aten_ops_stepN = [
        "torch.ops.aten.arange.start_step",
        "torch.ops.aten.permute_copy.default",
        "torch.ops.aten.index_put.default",
    ]
    int_exir_ops_step1 = [
        "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor",
        "executorch_exir_dialects_edge__ops_aten_cat_default",
    ]
    int_exir_ops_stepN = [
        "executorch_exir_dialects_edge__ops_aten_arange_start_step",
        "executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        "executorch_exir_dialects_edge__ops_aten_index_put_default",
    ]
    u55_not_supported = {
        "executorch_exir_dialects_edge__ops_aten_index_put_default": 1,
    }

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int,
        start: int | None,
        end: int | None,
        step: int,
    ):
        return x.slice_scatter(y, dim=dim, start=start, end=end, step=step)


input_t = Tuple[torch.Tensor, torch.Tensor, int, int | None, int | None, int]


@common.parametrize(
    "test_module", test_data_fp_step1 | test_data_fp_stepN | test_data_bf16
)
def test_slice_scatter_tosa_FP(test_module: input_t):
    pipeline = TosaPipelineFP[input_t](
        SliceScatter(),
        test_module(),
        aten_op=SliceScatter.fp_aten_op,
        exir_op=SliceScatter.fp_exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_module", test_data_fp_step1 | test_data_int_step1)
def test_slice_scatter_tosa_INT_step1(test_module: input_t):
    pipeline = TosaPipelineINT[input_t](
        SliceScatter(),
        test_module(),
        aten_op=SliceScatter.int_aten_ops_step1,
        exir_op=SliceScatter.int_exir_ops_step1,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_fp_stepN | test_data_int_stepN)
def test_slice_scatter_tosa_INT_stepN(test_module: input_t):
    pipeline = TosaPipelineINT[input_t](
        SliceScatter(),
        test_module(),
        aten_op=SliceScatter.int_aten_ops_stepN,
        exir_op=SliceScatter.int_exir_ops_stepN,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_fp_step1 | test_data_int_step1)
def test_slice_scatter_u55_INT_step1(test_module: input_t):
    # slice_scatter with unit-step is supported on U55
    pipeline = EthosU55PipelineINT[input_t](
        SliceScatter(),
        test_module(),
        aten_ops=SliceScatter.int_aten_ops_step1,
        exir_ops=SliceScatter.int_exir_ops_step1,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_fp_stepN | test_data_int_stepN)
def test_slice_scatter_u55_INT_stepN(test_module: input_t):
    # slice_scatter with non unit-step is not supported on U55
    pipeline = OpNotSupportedPipeline[input_t](
        SliceScatter(),
        test_module(),
        SliceScatter.u55_not_supported,
        quantize=True,
        u55_subset=True,
        n_expected_delegates=2,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_module", test_data_fp_step1 | test_data_int_step1)
def test_slice_scatter_u85_INT_step1(test_module: input_t):
    pipeline = EthosU85PipelineINT[input_t](
        SliceScatter(),
        test_module(),
        aten_ops=SliceScatter.int_aten_ops_step1,
        exir_ops=SliceScatter.int_exir_ops_step1,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_module", test_data_fp_stepN | test_data_int_stepN)
def test_slice_scatter_u85_INT_stepN(test_module: input_t):
    pipeline = EthosU85PipelineINT[input_t](
        SliceScatter(),
        test_module(),
        aten_ops=SliceScatter.int_aten_ops_stepN,
        exir_ops=SliceScatter.int_exir_ops_stepN,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize(
    "test_module",
    test_data_int_step1 | test_data_int_stepN | test_data_fp_step1 | test_data_fp_stepN,
    xfails={
        "rank2_step1_int8": "MLETORCH-1823: Fix quantized-node detection",
        "rank3_step2_int32": "MLETORCH-1823: Fix quantized-node detection",
    },
)
def test_slice_scatter_vgf_no_quant(test_module: input_t):
    pipeline = VgfPipeline[input_t](
        SliceScatter(),
        test_module(),
        aten_op=SliceScatter.fp_aten_op,
        exir_op=SliceScatter.fp_exir_op,
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_fp_step1 | test_data_int_step1)
def test_slice_scatter_vgf_quant_step1(test_module: input_t):
    pipeline = VgfPipeline[input_t](
        SliceScatter(),
        test_module(),
        aten_op=SliceScatter.int_aten_ops_step1,
        exir_op=SliceScatter.int_exir_ops_step1,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_fp_stepN | test_data_int_stepN)
def test_slice_scatter_vgf_quant_stepN(test_module: input_t):
    pipeline = VgfPipeline[input_t](
        SliceScatter(),
        test_module(),
        aten_op=SliceScatter.int_aten_ops_stepN,
        exir_op=SliceScatter.int_exir_ops_stepN,
        quantize=True,
    )
    pipeline.run()
