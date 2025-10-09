# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


class Amin(torch.nn.Module):
    input_t = Tuple[Tuple[torch.Tensor], int | Tuple[int], bool]
    aten_op = ["torch.ops.aten.amin"]

    def __init__(self, dim, keep_dims):
        self.dim = dim
        self.keep_dims = keep_dims
        super().__init__()

    def forward(self, x):
        if self.dim is None:
            return torch.amin(x, keepdim=self.keep_dims)
        else:
            return torch.amin(x, self.dim, self.keep_dims)

    test_data: Dict = {
        "rank_1_dim_0": lambda: ((torch.rand([10]),), 0, False),
        "rank_2_dim_1_keep_dims": lambda: ((torch.rand([2, 2]),), (1,), True),
        "rank_4_all_dim": lambda: ((torch.rand([1, 2, 5, 5]),), (0, 1, 2, 3), False),
        "rank_4_no_dim": lambda: ((torch.rand([1, 2, 5, 5]),), None, False),
        "rank_4_0,3_keep_dims": lambda: ((torch.rand([1, 2, 2, 2]),), (0, 3), True),
        "rank_4_mult_batches": lambda: ((torch.rand([2, 2, 2, 2]),), (0), True),
    }


class Min(torch.nn.Module):
    input_t = Tuple[Tuple[torch.Tensor], int]
    aten_op = ["torch.ops.aten.amin"]

    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def forward(self, x):
        x = torch.min(x, self.dim)
        return x[0]

    test_data: Dict = {
        "rank_1_dim_0": lambda: ((torch.rand([10]),), 0),
        "rank_2_dim_1": lambda: ((torch.rand([2, 2]),), 1),
        "rank_4_dim_2": lambda: ((torch.rand([2, 2, 2, 2]),), 2),
        "rank_4_dim_3": lambda: ((torch.rand([2, 2, 2, 2]),), 3),
    }


class MinWithIndex(torch.nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def forward(self, x):
        x, i = torch.min(x, self.dim)
        return x, i


@common.parametrize("test_data", Amin.test_data)
def test_amin_tosa_FP(test_data: Amin.input_t):
    data, dim, keep_dims = test_data()
    pipeline = TosaPipelineFP[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        Amin.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Amin.test_data)
def test_amin_tosa_INT(test_data: Amin.input_t):
    data, dim, keep_dims = test_data()
    pipeline = TosaPipelineINT[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        Amin.aten_op,
    )
    pipeline.run()


def test_amin_u55_INT_not_delegated():
    data, dim, keep_dims = Amin.test_data["rank_4_all_dim"]()
    pipeline = OpNotSupportedPipeline[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        {" executorch_exir_dialects_edge__ops_aten_amin_default": 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", Amin.test_data)
@common.XfailIfNoCorstone320
def test_amin_u85_INT(test_data: Amin.input_t):
    data, dim, keep_dims = test_data()
    pipeline = EthosU85PipelineINT[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        Amin.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Min.test_data)
def test_min_dim_tosa_FP_to_amin(test_data: Min.input_t):
    data, dim = test_data()
    pipeline = TosaPipelineFP[Min.input_t](Min(dim), data, "torch.ops.aten.min")
    pipeline.run()


@common.parametrize("test_data", Min.test_data)
def test_min_dim_tosa_INT_to_amin(test_data: Min.input_t):
    data, dim = test_data()
    module = Min(dim)
    pipeline = TosaPipelineINT[Min.input_t](module, data, "torch.ops.aten.amin")
    pipeline.run()


@pytest.mark.xfail(reason="MLETORCH-718 : Quantization of indices in arm_quantizer")
def test_min_dim_tosa_INT_not_delegated():
    data, dim = Min.test_data["rank_4_dim_3"]()
    pipeline = OpNotSupportedPipeline[Min.input_t](
        MinWithIndex(dim),
        data,
        {},
        quantize=True,
    )
    pipeline.run()


def test_min_dim_tosa_FP_not_delegated():
    data, dim = Min.test_data["rank_4_dim_3"]()
    pipeline = OpNotSupportedPipeline[Min.input_t](MinWithIndex(dim), data, {})
    pipeline.run()


@common.parametrize("test_data", Amin.test_data)
@common.SkipIfNoModelConverter
@pytest.mark.xfail(reason="MLETORCH-1410: Tensor dimension count not supported: 0")
def test_amin_vgf_FP(test_data: Amin.input_t):
    data, dim, keep_dims = test_data()
    pipeline = VgfPipeline[Amin.input_t](
        Amin(dim, keep_dims), data, Amin.aten_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", Amin.test_data)
@common.SkipIfNoModelConverter
@pytest.mark.xfail(reason="MLETORCH-1410: Tensor dimension count not supported: 0")
def test_amin_vgf_INT(test_data: Amin.input_t):
    data, dim, keep_dims = test_data()
    pipeline = VgfPipeline[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        Amin.aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", Min.test_data)
@common.SkipIfNoModelConverter
@pytest.mark.xfail(reason="MLETORCH-1410: Tensor dimension count not supported: 0")
def test_min_dim_vgf_FP_to_amin(test_data: Min.input_t):
    data, dim = test_data()
    pipeline = VgfPipeline[Min.input_t](
        Min(dim),
        data,
        "torch.ops.aten.min",
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Min.test_data)
@common.SkipIfNoModelConverter
@pytest.mark.xfail(reason="MLETORCH-1410: Tensor dimension count not supported: 0")
def test_min_dim_vgf_INT_to_amin(test_data: Min.input_t):
    data, dim = test_data()
    pipeline = VgfPipeline[Min.input_t](
        Min(dim),
        data,
        "torch.ops.aten.amin",
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
