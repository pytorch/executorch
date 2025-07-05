# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.linear.default"

input_t1 = Tuple[torch.Tensor]

test_data_suite_rank1 = {
    # (test_name, test_data, out_features, has_bias)
    "model_linear_rank1_zeros": lambda: (
        torch.zeros(10),
        15,
        True,
    ),
    "model_linear_rank1_ones": lambda: (
        torch.ones(10),
        15,
        False,
    ),
    "model_linear_rank1_negative_ones": lambda: (
        torch.ones(10) * (-1),
        20,
        True,
    ),
    "model_linear_rank1_rand": lambda: (
        torch.rand(10),
        10,
        True,
    ),
    "model_linear_rank1_negative_large_rand": lambda: (
        torch.rand(10) * (-100),
        30,
        False,
    ),
    "model_linear_rank1_large_randn": lambda: (
        torch.randn(15) * 100,
        20,
        True,
    ),
}

test_data_suite_rank4 = {
    # (test_name, test_data, out_features, has_bias)
    "model_linear_rank4_zeros": lambda: (
        torch.zeros(5, 10, 25, 20),
        30,
        True,
    ),
    "model_linear_rank4_ones": lambda: (
        torch.ones(5, 10, 25, 20),
        30,
        False,
    ),
    "model_linear_rank4_negative_ones": lambda: (
        torch.ones(5, 10, 25, 20) * (-1),
        30,
        True,
    ),
    "model_linear_rank4_rand": lambda: (
        torch.rand(5, 10, 25, 20),
        30,
        False,
    ),
    "model_linear_rank4_negative_large_rand": lambda: (
        torch.rand(5, 10, 25, 20) * (-100),
        30,
        True,
    ),
    "model_linear_rank4_large_randn": lambda: (
        torch.randn(5, 10, 25, 20) * 100,
        30,
        False,
    ),
}


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        return self.fc(x)


@common.parametrize("test_data", test_data_suite_rank1 | test_data_suite_rank4)
def test_linear_tosa_MI(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    pipeline = TosaPipelineMI[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness.
@common.parametrize("test_data", test_data_suite_rank1 | test_data_suite_rank4)
def test_linear_tosa_BI(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    pipeline = TosaPipelineBI[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_rank1)
@common.XfailIfNoCorstone300
def test_linear_u55_BI(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    EthosU55PipelineBI[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    ).run()


x_fail = {
    "model_linear_rank4_zeros": "AssertionError: Output 0 does not match reference output.",
    "model_linear_rank4_ones": "AssertionError: Output 0 does not match reference output.",
    "model_linear_rank4_negative_ones": "AssertionError: Output 0 does not match reference output.",
    "model_linear_rank4_rand": "AssertionError: Output 0 does not match reference output.",
    "model_linear_rank4_negative_large_rand": "AssertionError: Output 0 does not match reference output.",
    "model_linear_rank4_large_randn": "AssertionError: Output 0 does not match reference output.",
}


@common.parametrize(
    "test_data",
    test_data_suite_rank1 | test_data_suite_rank4,
    x_fail,
)
@common.XfailIfNoCorstone320
def test_linear_u85_BI(test_data: torch.Tensor):
    test_data, out_features, has_bias = test_data()
    in_features = test_data.shape[-1]
    EthosU85PipelineBI[input_t1](
        Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        ),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    ).run()
