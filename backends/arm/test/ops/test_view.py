# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the view op which changes the size of a Tensor without changing the underlying data.
#

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.view.default"

input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x,  Input y


class View(torch.nn.Module):

    needs_transpose_tests = {
        "rand_1d_neg": lambda: (torch.rand(100), (1, -1, 5, 2)),
        "rand_4d_neg": lambda: (torch.rand(10, 2, 1, 5), (1, -1, 5, 2)),
        "rand_4d_4d_small": lambda: (torch.rand(1, 2, 1, 9), (3, 1, 3, 2)),
        "rand_4d_4d": lambda: (torch.rand(2, 1, 1, 9), (3, 2, 3, 1)),
        "rand_4d_2d": lambda: (torch.rand(2, 50, 2, 1), (1, 200)),
        "rand_4d_3d": lambda: (torch.rand(2, 5, 2, 3), (1, 15, 4)),
        "rand_4d_1": lambda: (torch.rand(2, 1, 1, 9), (3, 1, 3, 2)),
        "rand_4d_2": lambda: (torch.rand(5, 10, 1, 1), (25, 2, 1, 1)),
        "rand_4d_2_4": lambda: (torch.rand(10, 2), (1, 1, 5, 4)),
        "rand_4d_2_4_big": lambda: (torch.rand(10, 10), (5, 1, 5, 4)),
        "rand_4d_4_4": lambda: (torch.rand(1, 1, 1, 10), (1, 1, 10, 1)),
        "rand_4d_4_4_big": lambda: (torch.rand(1, 1, 5, 10), (1, 1, 50, 1)),
        "rand_4d_4_3": lambda: (torch.rand(5, 10, 1, 1), (1, 25, 2)),
        "rand_4d_4_2": lambda: (torch.rand(2, 50, 1, 1), (1, 100)),
        "rand_4d_2_4_same": lambda: (torch.rand(2, 3, 2, 3), (2, 3, 3, 2)),
    }

    rank_product_too_large = {
        "rand_4d_large": lambda: (torch.rand(1, 49, 16, 128), (1, 16, 49, 128)),
    }

    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x: torch.Tensor):
        return x.view(self.new_shape)


@common.parametrize("test_data", View.needs_transpose_tests)
def test_view_tosa_FP(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = TosaPipelineFP[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
def test_view_tosa_INT(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = TosaPipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.XfailIfNoCorstone300
def test_view_u55_INT(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.SkipIfNoModelConverter
def test_view_vgf_FP(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = VgfPipeline[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.SkipIfNoModelConverter
def test_view_vgf_INT(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = VgfPipeline[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", View.rank_product_too_large)
@common.XfailIfNoCorstone300
def test_view_u55_INT_not_delegated(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        View(new_shape),
        (test_tensor,),
        {"executorch_exir_dialects_edge__ops_aten_view_copy": 1},
        n_expected_delegates=0,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_data", View.needs_transpose_tests)
@common.XfailIfNoCorstone320
def test_view_u85_INT(test_data: Tuple):
    test_tensor, new_shape = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        View(new_shape),
        (test_tensor,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()
