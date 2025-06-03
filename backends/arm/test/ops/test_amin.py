# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineBI,
    OpNotSupportedPipeline,
    TosaPipelineBI,
    TosaPipelineMI,
)


class Amin(torch.nn.Module):
    input_t = Tuple[Tuple[torch.Tensor], int | Tuple[int], bool]
    aten_op = ["torch.ops.aten.amin"]

    def __init__(self, dim, keep_dims):
        self.dim = dim
        self.keep_dims = keep_dims
        super().__init__()

    def forward(self, x):
        return torch.amin(x, self.dim, self.keep_dims)

    test_data: Dict[str, input_t] = {
        "rank_1_dim_0": lambda: ((torch.rand([10]),), 0, False),
        "rank_2_dim_1_keep_dims": lambda: ((torch.rand([2, 2]),), (1,), True),
        "rank_4_all_dim": lambda: ((torch.rand([1, 2, 5, 5]),), (0, 1, 2, 3), False),
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

    test_data: Dict[str, input_t] = {
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
def test_amin_tosa_MI(test_data: Amin.input_t):
    data, dim, keep_dims = test_data()
    pipeline = TosaPipelineMI[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        Amin.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Amin.test_data)
def test_amin_tosa_BI(test_data: Amin.input_t):
    data, dim, keep_dims = test_data()
    pipeline = TosaPipelineBI[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        Amin.aten_op,
    )
    pipeline.run()


def test_amin_u55_BI_not_delegated():
    data, dim, keep_dims = Amin.test_data["rank_4_all_dim"]()
    pipeline = OpNotSupportedPipeline[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        {" executorch_exir_dialects_edge__ops_aten_amin_default": 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


fvp_xfails = {"rank_4_mult_batches": "MLETORCH-517 : Multiple batches not supported"}


@common.parametrize("test_data", Amin.test_data, fvp_xfails)
@common.XfailIfNoCorstone320
def test_amin_u85_BI(test_data: Amin.input_t):
    data, dim, keep_dims = test_data()
    pipeline = EthosU85PipelineBI[Amin.input_t](
        Amin(dim, keep_dims),
        data,
        Amin.aten_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Min.test_data)
def test_min_dim_tosa_MI_to_amin(test_data: Min.input_t):
    data, dim = test_data()
    pipeline = TosaPipelineMI[Min.input_t](Min(dim), data, "torch.ops.aten.min")
    pipeline.run()


@common.parametrize("test_data", Min.test_data)
def test_min_dim_tosa_BI_to_amin(test_data: Min.input_t):
    data, dim = test_data()
    module = Min(dim)
    pipeline = TosaPipelineBI[Min.input_t](module, data, "torch.ops.aten.amin")
    pipeline.run()


@pytest.mark.xfail(reason="MLETORCH-718 : Quantization of indices in arm_quantizer")
def test_min_dim_tosa_BI_not_delegated():
    data, dim = Min.test_data["rank_4_dim_3"]()
    pipeline = OpNotSupportedPipeline[Min.input_t](
        MinWithIndex(dim),
        data,
        {},
        quantize=True,
    )
    pipeline.run()


def test_min_dim_tosa_MI_not_delegated():
    data, dim = Min.test_data["rank_4_dim_3"]()
    pipeline = OpNotSupportedPipeline[Min.input_t](MinWithIndex(dim), data, {})
    pipeline.run()
