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


class Amax(torch.nn.Module):
    input_t = Tuple[Tuple[torch.Tensor], int | Tuple[int], bool]
    aten_op = ["torch.ops.aten.amax"]

    def __init__(self, dim, keep_dims):
        self.dim = dim
        self.keep_dims = keep_dims
        super().__init__()

    def forward(self, x):
        return torch.amax(x, self.dim, self.keep_dims)

    test_data: Dict[str, input_t] = {
        "rank_1_dim_0": lambda: ((torch.rand([10]),), 0, False),
        "rank_2_dim_1_keep_dims": lambda: ((torch.rand([2, 2]),), (1,), True),
        "rank_4_all_dim": lambda: ((torch.rand([1, 2, 5, 5]),), (0, 1, 2, 3), False),
        "rank_4_0,3_keep_dims": lambda: ((torch.rand([1, 2, 2, 2]),), (0, 3), True),
        "rank_4_mult_batches": lambda: ((torch.rand([2, 2, 2, 2]),), (0), True),
    }


class Max(torch.nn.Module):
    input_t = Tuple[Tuple[torch.Tensor], int]
    aten_op = ["torch.ops.aten.amax"]

    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def forward(self, x):
        x = torch.max(x, self.dim, False)
        return x[0]

    test_data: Dict[str, input_t] = {
        "rank_1_dim_0": lambda: ((torch.rand([10]),), 0),
        "rank_2_dim_1": lambda: ((torch.rand([2, 2]),), 1),
        "rank_4_dim_2": lambda: ((torch.rand([2, 2, 2, 2]),), 2),
        "rank_4_dim_3": lambda: ((torch.rand([2, 2, 2, 2]),), 3),
    }


class MaxWithIndex(torch.nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def forward(self, x):
        x, i = torch.max(x, self.dim)
        return x, i


@common.parametrize("test_data", Amax.test_data)
def test_amax_tosa_FP(test_data: Amax.input_t):
    data, dim, keep_dims = test_data()
    pipeline = TosaPipelineFP[Amax.input_t](Amax(dim, keep_dims), data, Amax.aten_op)
    pipeline.run()


@common.parametrize("test_data", Amax.test_data)
def test_amax_tosa_INT(test_data: Amax.input_t):
    data, dim, keep_dims = test_data()
    pipeline = TosaPipelineINT[Amax.input_t](Amax(dim, keep_dims), data, Amax.aten_op)
    pipeline.run()


def test_amax_u55_INT_not_delegated():
    data, dim, keep_dims = Amax.test_data["rank_4_all_dim"]()
    pipeline = OpNotSupportedPipeline[Amax.input_t](
        Amax(dim, keep_dims),
        data,
        {" executorch_exir_dialects_edge__ops_aten_amax_default": 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", Amax.test_data)
@common.XfailIfNoCorstone320
def test_amax_u85_INT(test_data: Amax.input_t):
    data, dim, keep_dims = test_data()
    pipeline = EthosU85PipelineINT[Amax.input_t](
        Amax(dim, keep_dims),
        data,
        Amax.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Max.test_data)
def test_max_dim_tosa_FP_to_amax(test_data: Max.input_t):
    data, dim = test_data()
    pipeline = TosaPipelineFP[Max.input_t](Max(dim), data, "torch.ops.aten.max")
    pipeline.run()


@common.parametrize("test_data", Max.test_data)
def test_max_dim_tosa_INT_to_amax(test_data: Max.input_t):
    data, dim = test_data()
    module = Max(dim)
    pipeline = TosaPipelineINT[Max.input_t](module, data, "torch.ops.aten.amax")
    pipeline.run()


@pytest.mark.xfail(reason="MLETORCH-718 : Quantization of indices in arm_quantizer")
def test_max_dim_tosa_INT_not_delegated():
    data, dim = Max.test_data()["rank_4_dim_3"]()
    pipeline = OpNotSupportedPipeline[Max.input_t](
        MaxWithIndex(dim), data, {}, quantize=True
    )
    pipeline.run()


def test_max_dim_tosa_FP_not_delegated():
    data, dim = Max.test_data["rank_4_dim_3"]()
    pipeline = OpNotSupportedPipeline[Max.input_t](MaxWithIndex(dim), data, {})
    pipeline.run()


@common.parametrize("test_data", Amax.test_data)
@common.SkipIfNoModelConverter
def test_amax_vgf_no_quant(test_data: Amax.input_t):
    data, dim, keep_dims = test_data()
    module = Amax(dim, keep_dims)
    pipeline = VgfPipeline[Amax.input_t](
        module,
        data,
        Amax.aten_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", Amax.test_data)
@common.SkipIfNoModelConverter
def test_amax_vgf_quant(test_data: Amax.input_t):
    data, dim, keep_dims = test_data()
    module = Amax(dim, keep_dims)
    pipeline = VgfPipeline[Amax.input_t](
        module,
        data,
        Amax.aten_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", Max.test_data)
@common.SkipIfNoModelConverter
def test_max_dim_vgf_no_quant_to_amax(test_data: Max.input_t):
    data, dim = test_data()
    pipeline = VgfPipeline[Max.input_t](
        Max(dim),
        data,
        "torch.ops.aten.max",
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", Max.test_data)
@common.SkipIfNoModelConverter
def test_max_dim_vgf_quant_to_amax(test_data: Max.input_t):
    data, dim = test_data()
    pipeline = VgfPipeline[Max.input_t](
        Max(dim),
        data,
        "torch.ops.aten.amax",
        quantize=True,
    )
    pipeline.run()
