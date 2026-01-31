# Copyright 2026 Arm Limited and/or its affiliates.
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


class UnfoldCopy(torch.nn.Module):
    aten_op = "torch.ops.aten.unfold_copy.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_unfold_copy_default"

    def forward(self, input_: torch.Tensor, dim_: int, size_: int, step_: int):
        return torch.ops.aten.unfold_copy.default(input_, dim_, size_, step_)


input_params = Tuple[torch.Tensor, int, int, int]

# ---- FP profile: only float inputs ----
test_data_fp: dict[str, input_params] = {
    # 1D: [T] -> unfold dim=0 => [U, C]
    "test_fp32_1d_dim0": (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),  # [T=5]
        0,
        3,
        2,  # U=(5-3)//2+1=2 -> [U=2, C=3]
    ),
    # 2D: [B, T] -> unfold dim=1 => [B, U, C]
    "test_fp32_2d_dim1": (
        torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5]],
            dtype=torch.float32,
        ),  # [B=2, T=5]
        1,
        3,
        2,  # U=(5-3)//2+1=2 -> [B=2, U=2, C=3]
    ),
    # 3D: [B, T, F] -> unfold dim=-1 => [B, T, U, C]
    "test_fp32_3d_dim_neg1": (
        torch.randn(2, 6, 4, dtype=torch.float32),  # [B=2, T=6, F=4]
        -1,
        3,
        1,  # U=(4-3)//1+1=2 -> [B=2, T=6, U=2, C=3]
    ),
    # 4D: [B, T, N, H] -> unfold dim=1 => [B, U, N, H, C]
    "test_fp32_4d_dim1": (
        torch.randn(2, 6, 3, 4, dtype=torch.float32),  # [B=2, T=6, N=3, H=4]
        1,
        3,
        2,  # U=(6-3)//2+1=2 -> [B=2, U=2, N=3, H=4, C=3]
    ),
    # 4D: [B, T, N, H] -> unfold dim=-1 => [B, T, N, U, C]
    "test_fp32_4d_dim_neg1": (
        torch.randn(2, 6, 3, 4, dtype=torch.float32),  # [B=2, T=6, N=3, H=4]
        -1,
        3,
        1,  # U=(4-3)//1+1=2 -> [B=2, T=6, N=3, U=2, C=3]
    ),
}

# ---- INT profile: integer inputs + bool ----
test_data_int: dict[str, input_params] = {
    # int8 1D: [T] -> unfold dim=0 => [U, C]
    "test_int8_1d_dim0": (
        torch.randint(-5, 5, size=(10,), dtype=torch.int8),  # [T=10]
        0,
        4,
        3,  # U=(10-4)//3+1=3 -> [U=3, C=4]
    ),
    # bool 1D: [T] -> unfold dim=-1 => [U, C]
    "test_bool_1d_dim_neg1": (
        torch.tensor([True, False, True, True], dtype=torch.bool),  # [T=4]
        -1,
        3,
        1,  # U=(4-3)//1+1=2 -> [U=2, C=3]
    ),
    # bool 2D: [B, T] -> unfold dim=0 => [U, T, C]
    "test_bool_2d_dim0": (
        torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
        ),  # [B=2, T=3]
        0,
        2,
        1,  # U=(2-2)//1+1=1 -> [U=1, T=3, C=2]
    ),
    # int8 3D: [B, T, F] -> unfold dim=1 => [B, U, F, C]
    "test_int8_3d_dim1": (
        torch.randint(-5, 5, size=(2, 8, 5), dtype=torch.int8),  # [B=2, T=8, F=5]
        1,
        4,
        2,  # U=(8-4)//2+1=3 -> [B=2, U=3, F=5, C=4]
    ),
    # int8 3D: [B, T, F] -> unfold dim=-1 => [B, T, U, C]
    "test_int8_3d_dim_neg1": (
        torch.randint(-5, 5, size=(2, 8, 5), dtype=torch.int8),  # [B=2, T=8, F=5]
        -1,
        3,
        2,  # U=(5-3)//2+1=2 -> [B=2, T=8, U=2, C=3]
    ),
    # int32 4D: [B, T, N, H] -> unfold dim=-1 => [B, T, N, U, C]
    "test_int32_4d_dim_neg1": (
        torch.randint(
            -50, 50, size=(2, 7, 2, 3), dtype=torch.int32
        ),  # [B=2, T=7, N=2, H=3]
        -1,
        2,
        1,  # U=(3-2)//1+1=2 -> [B=2, T=7, N=2, U=2, C=2]
    ),
}


@common.parametrize("test_data", test_data_fp)
def test_unfold_copy_tosa_FP(test_data: input_params):
    pipeline = TosaPipelineFP[input_params](
        UnfoldCopy(),
        test_data,
        aten_op=UnfoldCopy.aten_op,
        exir_op=UnfoldCopy.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_unfold_copy_tosa_INT(test_data: input_params):
    pipeline = TosaPipelineINT[input_params](
        UnfoldCopy(),
        test_data,
        aten_op=UnfoldCopy.aten_op,
        exir_op=UnfoldCopy.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
@common.XfailIfNoCorstone300
def test_unfold_copy_u55_INT(test_data: input_params):
    # Gather op is not supported on U55
    pipeline = OpNotSupportedPipeline[input_params](
        UnfoldCopy(),
        test_data,
        {UnfoldCopy.exir_op: 1},
        quantize=True,
        u55_subset=True,
        n_expected_delegates=0,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_int | test_data_fp,
    xfails={
        "test_int8_3d_dim_neg1": "MLETORCH-1732: rand test fails",
        "test_int32_4d_dim_neg1": "MLETORCH-1732: rand test fails",
        "test_fp32_3d_dim_neg1": "MLETORCH-1732: rand test fails",
        "test_fp32_4d_dim_neg1": "MLETORCH-1732: rand test fails",
    },
)
@common.XfailIfNoCorstone320
def test_unfold_copy_u85_INT(test_data: input_params):
    pipeline = EthosU85PipelineINT[input_params](
        UnfoldCopy(),
        test_data,
        aten_ops=UnfoldCopy.aten_op,
        exir_ops=UnfoldCopy.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_int)
@common.SkipIfNoModelConverter
def test_unfold_copy_vgf_no_quant(test_data: input_params):
    pipeline = VgfPipeline[input_params](
        UnfoldCopy(),
        test_data,
        aten_op=UnfoldCopy.aten_op,
        exir_op=UnfoldCopy.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_int)
@common.SkipIfNoModelConverter
def test_unfold_copy_vgf_quant(test_data: input_params):
    pipeline = VgfPipeline[input_params](
        UnfoldCopy(),
        test_data,
        aten_op=UnfoldCopy.aten_op,
        exir_op=UnfoldCopy.exir_op,
        quantize=True,
    )
    pipeline.run()
