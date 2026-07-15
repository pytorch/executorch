# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
)

aten_op = "torch.ops.aten.argmax.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_argmax_default"
input_t = Tuple[torch.Tensor]


class Argmax(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return torch.argmax(x, dim=self.dim).to(torch.int32)

    test_data: dict[str, Tuple[input_t, int]] = {
        "rank_1_dim_0": lambda: ((torch.rand(10),), 0),
        "rank_2_dim_1": lambda: ((torch.rand(2, 5),), 1),
        "rank_4_dim_2": lambda: ((torch.rand(1, 3, 4, 5),), 2),
        "rank_4_dim_3": lambda: ((torch.rand(1, 3, 4, 5),), 3),
        "rank_4_dim_neg1": lambda: ((torch.rand(1, 3, 4, 5),), -1),
    }

    test_data_fp16: dict[str, Tuple[input_t, int]] = {
        "rank_2_dim_1_fp16": lambda: ((torch.rand(2, 5, dtype=torch.float16),), 1),
    }

    test_data_bf16: dict[str, Tuple[input_t, int]] = {
        "rank_2_dim_1_bf16": lambda: ((torch.rand(2, 5, dtype=torch.bfloat16),), 1),
    }

    test_data_int: dict[str, Tuple[input_t, int]] = {
        "rank_1_dim_0_int8": lambda: (
            (torch.randint(-128, 127, (10,), dtype=torch.int8),),
            0,
        ),
        "rank_2_dim_1_int8": lambda: (
            (torch.randint(-128, 127, (2, 5), dtype=torch.int8),),
            1,
        ),
        "rank_4_dim_2_int8": lambda: (
            (torch.randint(-128, 127, (1, 3, 4, 5), dtype=torch.int8),),
            2,
        ),
        "rank_4_dim_3_int8": lambda: (
            (torch.randint(-128, 127, (1, 3, 4, 5), dtype=torch.int8),),
            3,
        ),
    }


class ArgmaxAll(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.argmax(x)


class ArgmaxKeepDim(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.argmax(x, dim=1, keepdim=True)


class ArgmaxInt32(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.argmax(x, dim=1).to(torch.int32)


@common.parametrize(
    "test_data", Argmax.test_data | Argmax.test_data_fp16 | Argmax.test_data_bf16
)
def test_argmax_tosa_FP(test_data: Tuple[input_t, int]):
    data, dim = test_data()
    pipeline = TosaPipelineFP[input_t](
        Argmax(dim),
        data,
        aten_op,
        exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.count_tosa_ops({"ARGMAX": 1})
    pipeline.run()


def test_argmax_all_tosa_FP_not_delegated():
    pipeline = OpNotSupportedPipeline[input_t](
        ArgmaxAll(),
        (torch.rand(2, 5),),
        {exir_op: 1},
    )
    pipeline.run()


def test_argmax_keepdim_tosa_FP_not_delegated():
    pipeline = OpNotSupportedPipeline[input_t](
        ArgmaxKeepDim(),
        (torch.rand(2, 5),),
        {exir_op: 1},
    )
    pipeline.run()


def test_argmax_int32_tosa_FP_not_delegated():
    pipeline = OpNotSupportedPipeline[input_t](
        ArgmaxInt32(),
        (torch.randint(0, 10, (2, 5), dtype=torch.int32),),
        {exir_op: 1},
    )
    pipeline.run()


@common.parametrize("test_data", Argmax.test_data_int)
def test_argmax_tosa_INT(test_data: Tuple[input_t, int]):
    data, dim = test_data()
    pipeline = TosaPipelineINT[input_t](
        Argmax(dim),
        data,
        aten_op,
        exir_op,
    )
    pipeline.count_tosa_ops({"ARGMAX": 1})
    pipeline.run()
