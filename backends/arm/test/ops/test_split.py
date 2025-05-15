# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

exir_op = "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default"
input_t1 = Tuple[torch.Tensor]  # Input x


class Split(torch.nn.Module):

    test_data = {
        "split_1d_2_size_0_dim": lambda: (torch.rand(10), 2, 0),
        "split_2d_3_size_1_dim": lambda: (torch.rand(10, 10), 3, 1),
        "split_2d_2_size_4_dim": lambda: (torch.rand(10, 10), 4, -1),
        "split_4d_2_size_2_dim": lambda: (torch.rand(4, 4, 4, 4), 2, 0),
    }

    test_data_list = {
        "split_3d_2_sizes_dim": lambda: (torch.rand(10, 15, 10), [2, 2, 11], 1),
        "split_4d_2_sizes_dim_neg": lambda: (torch.rand(4, 4, 4, 4), [1, 1, 1, 1], -2),
    }

    def forward(
        self, x: torch.Tensor, split_size_or_sections: int | list[int], dim: int
    ):
        return x.split(split_size=split_size_or_sections, dim=dim)


class SplitWithSizes(torch.nn.Module):
    def forward(self, x: torch.Tensor, split_sizes: list[int], dim: int):
        return x.split_with_sizes(split_sizes=split_sizes, dim=dim)


class SplitSingleOut(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, split_size_or_sections: int | list[int], dim: int
    ):
        return x.split(split_size=split_size_or_sections, dim=dim)[1]


class SplitTwoOut(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, split_size_or_sections: int | list[int], dim: int
    ):
        return x.split(split_size=split_size_or_sections, dim=dim)[1:3]


@common.parametrize(
    "test_data",
    (Split.test_data | Split.test_data_list),
)
def test_split_with_sizes_tosa_MI(test_data: input_t1):

    pipeline = TosaPipelineMI[input_t1](
        Split(),
        test_data(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Split.test_data_list)
def test_split_with_sizes_tosa_MI_2(test_data: input_t1):

    pipeline = TosaPipelineMI[input_t1](
        SplitWithSizes(),
        test_data(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    (Split.test_data | Split.test_data_list),
)
def test_split_with_sizes_tosa_MI_one_out(test_data: input_t1):

    pipeline = TosaPipelineMI[input_t1](
        SplitSingleOut(),
        test_data(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    (Split.test_data | Split.test_data_list),
)
def test_split_with_sizes_tosa_BI(test_data: input_t1):

    pipeline = TosaPipelineBI[input_t1](
        Split(),
        test_data(),
        aten_op=[],
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    (Split.test_data | Split.test_data_list),
)
def test_split_with_sizes_u55_BI(test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        Split(),
        test_data(),
        aten_ops=[],
        exir_ops=exir_op,
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    (Split.test_data | Split.test_data_list),
)
def test_split_with_sizes_u85_BI(test_data: input_t1):

    pipeline = EthosU85PipelineBI[input_t1](
        Split(),
        test_data(),
        aten_ops=[],
        exir_ops=exir_op,
        run_on_fvp=False,
    )
    pipeline.run()
