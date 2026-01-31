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


class IndexSelect(torch.nn.Module):
    aten_op = "torch.ops.aten.index_select.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_index_select_default"

    def forward(self, input_: torch.Tensor, dim: int, index_: torch.Tensor):
        return torch.index_select(input_, dim=dim, index=index_)


input_params = Tuple[torch.Tensor, int, torch.Tensor]

# ---- FP profile: only float inputs ----
test_data_fp: dict[str, input_params] = {
    # Rank-1: [K] -> index_select dim=0 => [W]
    "test_fp32_rank1_dim0": (
        torch.randn(6, dtype=torch.float32),  # [K=6]
        0,
        torch.tensor([1, 4, 5], dtype=torch.int32),  # [W=3]
    ),
    # Rank-2: [K, C] -> index_select dim=0 => [W, C]
    "test_fp32_rank2_dim0": (
        torch.randn(4, 3, dtype=torch.float32),  # [K=4, C=3]
        0,
        torch.tensor([1, 3], dtype=torch.int32),  # [W=2]
    ),
    # Rank-3: [N, K, C] -> index_select dim=-1 => [N, K, W]
    "test_fp32_rank3_dim_neg1": (
        torch.randn(2, 4, 3, dtype=torch.float32),  # [N=2, K=4, C=3]
        -1,
        torch.tensor([2, 0], dtype=torch.int32),  # [W=2]
    ),
    # Rank-3: [N, K, C] -> index_select dim=1 => [N, W, C]
    "test_fp32_rank3_dim1": (
        torch.randn(2, 4, 3, dtype=torch.float32),  # [N=2, K=4, C=3]
        1,
        torch.tensor([1, 3], dtype=torch.int32),  # [W=2]
    ),
    # Rank-4: [A, B, K, C] -> index_select dim=2 => [A, B, W, C]
    "test_fp32_rank4_dim2": (
        torch.randn(2, 3, 4, 5, dtype=torch.float32),  # [A=2, B=3, K=4, C=5]
        2,
        torch.tensor([3, 1], dtype=torch.int32),  # [W=2]
    ),
}

# ---- INT profile: integer inputs + bool ----
test_data_int: dict[str, input_params] = {
    # Rank-1 int8: [K] -> index_select dim=0 => [W]
    "test_int8_rank1_dim0": (
        torch.randint(-6, 6, size=(6,), dtype=torch.int8),  # [K=6]
        0,
        torch.tensor([5, 0, 2], dtype=torch.int32),  # [W=3]
    ),
    # Rank-2 bool: [K, C] -> index_select dim=0 => [W, C]
    "test_bool_rank2_dim0": (
        torch.randint(0, 2, size=(3, 2), dtype=torch.int8).to(torch.bool),  # [K=3, C=2]
        0,
        torch.tensor([2, 0], dtype=torch.int32),  # [W=2]
    ),
    # Rank-3 int8: [N, K, C] -> index_select dim=1 => [N, W, C]
    "test_int8_rank3_dim1": (
        torch.randint(-5, 5, size=(2, 7, 4), dtype=torch.int8),  # [N=2, K=7, C=4]
        1,
        torch.tensor([0, 6, 3], dtype=torch.int32),  # [W=3]
    ),
    # Rank-4 int32: [A, B, K, C] -> index_select dim=2 => [A, B, W, C]
    "test_int32_rank4_dim2": (
        torch.randint(
            -20, 20, size=(2, 3, 5, 4), dtype=torch.int32
        ),  # [A=2, B=3, K=5, C=4]
        2,
        torch.tensor([4, 1], dtype=torch.int32),  # [W=2]
    ),
}


@common.parametrize("test_data", test_data_fp)
def test_index_select_tosa_FP(test_data: input_params):
    pipeline = TosaPipelineFP[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_index_select_tosa_INT(test_data: input_params):
    # INT profile runs quantized, so we test both int inputs and float inputs here.
    pipeline = TosaPipelineINT[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_index_select_u55_INT_not_delegated(test_data: input_params):
    pipeline = OpNotSupportedPipeline[input_params](
        IndexSelect(),
        test_data,
        {IndexSelect.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_index_select_u85_INT(test_data: input_params):
    pipeline = EthosU85PipelineINT[input_params](
        IndexSelect(),
        test_data,
        aten_ops=IndexSelect.aten_op,
        exir_ops=IndexSelect.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_int)
@common.SkipIfNoModelConverter
def test_index_select_vgf_no_quant(test_data: input_params):
    pipeline = VgfPipeline[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_int)
@common.SkipIfNoModelConverter
def test_index_select_vgf_quant(test_data: input_params):
    pipeline = VgfPipeline[input_params](
        IndexSelect(),
        test_data,
        aten_op=IndexSelect.aten_op,
        exir_op=IndexSelect.exir_op,
        quantize=True,
    )
    pipeline.run()
