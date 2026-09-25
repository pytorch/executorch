# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.exir.dialects._ops import ops as exir_ops


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
test_data_fp8: dict[str, input_params] = {
    # 2D: [B, T] -> unfold dim=1 => [B, U, C]
    "test_fp8e4m3_2d_dim1": (
        torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5]],
            dtype=torch.float32,
        ).to(
            torch.float8_e4m3fn
        ),  # [B=2, T=5]
        1,
        3,
        2,  # U=(5-3)//2+1=2 -> [B=2, U=2, C=3]
        "fp8e4m3",
    ),
    # 3D: [B, T, F] -> unfold dim=-1 => [B, T, U, C]
    "test_fp8e5m2_3d_dim_neg1": (
        torch.randn(2, 6, 4, dtype=torch.float32).to(
            torch.float8_e5m2
        ),  # [B=2, T=6, F=4]
        -1,
        3,
        1,  # U=(4-3)//1+1=2 -> [B=2, T=6, U=2, C=3]
        "fp8e5m2",
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

test_data_bf16: dict[str, input_params] = {
    "test_bf16_2d_dim1": (
        torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5]],
            dtype=torch.bfloat16,
        ),  # [B=2, T=5]
        1,
        3,
        2,  # U=(5-3)//2+1=2 -> [B=2, U=2, C=3]
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


@common.parametrize("test_data", test_data_bf16)
def test_unfold_copy_tosa_FP_bf16(test_data: input_params):
    pipeline = TosaPipelineFP[input_params](
        UnfoldCopy(),
        test_data,
        aten_op=UnfoldCopy.aten_op,
        exir_op=UnfoldCopy.exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp8)
def test_unfold_copy_tosa_FP_fp8(test_data):
    input_, dim_, size_, step_, tosa_extension = test_data
    pipeline = TosaPipelineFP[input_params](
        UnfoldCopy(),
        (input_, dim_, size_, step_),
        aten_op=UnfoldCopy.aten_op,
        exir_op=UnfoldCopy.exir_op,
        compare_tosa_ref_model_outputs=False,
        tosa_extensions=[tosa_extension],
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


test_data_u55 = {
    "rank1_dim0": test_data_fp["test_fp32_1d_dim0"],
    "rank2_dim1": test_data_fp["test_fp32_2d_dim1"],
    "rank2_max_windows": (torch.rand(1, 17), 1, 2, 1),
    "rank3_dim1": test_data_int["test_int8_3d_dim1"],
    "rank3_dim_neg1": test_data_fp["test_fp32_3d_dim_neg1"],
}


@common.parametrize("test_data", test_data_u55)
@common.XfailIfNoCorstone300
def test_unfold_copy_u55_INT(test_data: input_params):
    pipeline = EthosU55PipelineINT[input_params](
        UnfoldCopy(),
        test_data,
        aten_ops=[],
        exir_ops=[],
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_unfold_copy_u55_INT_a16w8():
    pipeline = EthosU55PipelineINT[input_params](
        UnfoldCopy(),
        test_data_fp["test_fp32_2d_dim1"],
        aten_ops=[],
        exir_ops=[],
        a16w8_quantization=True,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    {
        "bool": test_data_int["test_bool_1d_dim_neg1"],
        "too_many_windows": (torch.rand(1, 20), 1, 2, 1),
    },
)
def test_unfold_copy_u55_INT_not_delegated(test_data: input_params):
    pipeline = OpNotSupportedPipeline[input_params](
        UnfoldCopy(),
        test_data,
        {UnfoldCopy.exir_op: 1},
        quantize=True,
        u55_subset=True,
        n_expected_delegates=0,
    )
    pipeline.run()


def test_unfold_copy_u55_INT_symbolic_dim_not_delegated():
    length = torch.export.Dim("length", min=4, max=12)
    tester = ArmTester(
        UnfoldCopy(),
        (torch.rand(2, 6), 1, 3, 1),
        common.get_u55_compile_spec(),
        dynamic_shapes={
            "input_": {1: length},
            "dim_": None,
            "size_": None,
            "step_": None,
        },
    )
    tester.quantize().export().to_edge().partition()

    targets = {
        node.target
        for node in tester.stages[tester.cur].artifact.exported_program().graph.nodes
    }
    assert exir_ops.edge.aten.unfold_copy.default in targets
    assert torch.ops.higher_order.executorch_call_delegate not in targets


@common.parametrize(
    "test_data",
    test_data_int | test_data_fp,
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


@common.parametrize("test_data", test_data_fp | test_data_bf16 | test_data_int)
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
