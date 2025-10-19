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
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
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
def test_permute_tosa_FP(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = TosaPipelineFP[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_permute_tosa_INT(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = TosaPipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={"rank_4_3": "MLETORCH-955 : Permutation numerical diff for u55"},
)
@common.XfailIfNoCorstone300
def test_permute_u55_INT(test_data):
    test_data, dims = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops="executorch_exir_dialects_edge__ops_aten_permute_copy_default",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_permute_u85_INT(test_data: torch.Tensor):
    test_data, dims = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_ops="executorch_exir_dialects_edge__ops_aten_permute_copy_default",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_permute_vgf_FP(test_data):
    test_data, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_permute_vgf_INT(test_data):
    test_data, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        SimplePermute(dims=dims),
        (test_data,),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
