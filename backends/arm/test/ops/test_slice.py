# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
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

aten_op = "torch.ops.aten.slice.Tensor"
exir_op = "executorch_exir_dialects_edge__ops_aten_slice_copy"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    "ones_neg_3": lambda: (torch.ones(10), [(3, -3)]),
    "ones_neg_8": lambda: (torch.ones(10), [(-8, 3)]),
    "ones_slice_2": lambda: (torch.ones(10, 10), [(1, 3), (3, None)]),
    "ones_slice_3": lambda: (torch.ones(10, 10, 10), [(0, 7), (0, None), (0, 8)]),
    "ones_slice_4": lambda: (
        torch.ones((1, 12, 10, 10)),
        [(None, None), (None, 5), (3, 5), (4, 10)],
    ),
}


class Slice(torch.nn.Module):

    def forward(self, x: torch.Tensor, s: list[tuple[int, int]]):
        slices = [slice(*i) for i in s]
        return x[slices]


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](Slice(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_INT_nchw(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_tosa_INT_nhwc(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_u55_INT(test_data: torch.Tensor):
    pipeline = EthosU55PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_slice_tensor_u85_INT(test_data: torch.Tensor):
    pipeline = EthosU85PipelineINT[input_t1](
        Slice(),
        test_data(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_slice_tensor_vgf_FP(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_slice_tensor_vgf_INT(test_data: torch.Tensor):
    pipeline = VgfPipeline[input_t1](
        Slice(),
        test_data(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
