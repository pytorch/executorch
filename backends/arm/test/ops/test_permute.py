# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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
from torchvision.ops import Permute

input_t1 = Tuple[torch.Tensor]  # Input x

aten_op = "torch.ops.aten.permute.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_permute_default"

test_data_suite = {
    # (test_name,test_data,dims)
    "rank_2": lambda: (torch.rand(10, 10), [1, 0]),
    "rank_3": lambda: (torch.rand(10, 10, 10), [2, 0, 1]),
    "rank_3_2": lambda: (torch.rand(10, 10, 10), [1, 2, 0]),
    "rank_4": lambda: (torch.rand(1, 5, 1, 10), [0, 2, 3, 1]),
    "rank_4_2": lambda: (torch.rand(1, 2, 5, 10), [1, 0, 2, 3]),
    "rank_4_3": lambda: (torch.rand(1, 10, 10, 5), [2, 0, 1, 3]),
}


class SimplePermute(torch.nn.Module):

    def __init__(self, dims: list[int]):
        super().__init__()

        self.permute = Permute(dims=dims)

    def forward(self, x):
        return self.permute(x)


@common.parametrize("test_data", test_data_suite)
def test_permute_tosa_MI(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = TosaPipelineMI[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_permute_tosa_BI(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = TosaPipelineBI[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
        symmetric_io_quantization=True,
    )
    pipeline.run()


x_fails = {
    "rank_4_2": "AssertionError: Output 0 does not match reference output.",
    "rank_4_3": "AssertionError: Output 0 does not match reference output.",
}


@common.parametrize("test_data", test_data_suite, x_fails)
@common.XfailIfNoCorstone300
def test_permute_u55_BI(test_data):
    test_data, dims = test_data()
    pipeline = EthosU55PipelineBI[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops="executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()


# Fails since on FVP since N > 1 is not supported. MLETORCH-517
@common.parametrize("test_data", test_data_suite, x_fails)
@common.XfailIfNoCorstone320
def test_permute_u85_BI(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = EthosU85PipelineBI[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops="executorch_exir_dialects_edge__ops_aten_permute_copy_default",
        run_on_fvp=True,
        symmetric_io_quantization=True,
    )
    pipeline.run()
