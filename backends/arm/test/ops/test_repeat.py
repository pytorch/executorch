# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the repeat op which copies the data of the input tensor (possibly with new data format)
#


from typing import Sequence, Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x, Input y


"""Tests Tensor.repeat for different ranks and dimensions."""


class Repeat(torch.nn.Module):
    aten_op = "torch.ops.aten.repeat.default"

    def __init__(self, multiples: Sequence[int]):
        super().__init__()
        self.multiples = multiples

    def forward(self, x: torch.Tensor):
        return x.repeat(self.multiples)


class RepeatInterleaveInt(torch.nn.Module):
    aten_op = "torch.ops.aten.repeat_interleave.self_int"

    def __init__(self, repeats: int, dim: int):
        super().__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.repeat_interleave(self.repeats, self.dim)


test_data_suite = {
    # test_name : lambda: (module, test_data)
    "1_x_1": lambda: (Repeat((2,)), (torch.randn(3),)),
    "2_x_2": lambda: (Repeat((2, 1)), (torch.randn(3, 4),)),
    "4_x_4": lambda: (Repeat((1, 2, 3, 4)), (torch.randn(1, 1, 2, 2),)),
    "1_x_2": lambda: (Repeat((2, 2)), (torch.randn(3),)),
    "1_x_3": lambda: (Repeat((1, 2, 3)), (torch.randn(3),)),
    "2_x_3": lambda: (Repeat((2, 2, 2)), (torch.randn((3, 3)),)),
    "1_x_4": lambda: (Repeat((2, 1, 2, 4)), (torch.randn((3, 3, 3)),)),
    "interleave_int_3_x_1": lambda: (RepeatInterleaveInt(3, 1), (torch.randn(3, 4),)),
}


@common.parametrize("test_data", test_data_suite)
def test_repeat_tosa_FP(test_data: Tuple):
    module, test_data = test_data()
    pipeline = TosaPipelineFP[input_t1](
        module,
        test_data,
        module.aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_repeat_tosa_INT(test_data: Tuple):
    module, test_data = test_data()
    pipeline = TosaPipelineINT[input_t1](
        module,
        test_data,
        module.aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_repeat_u55_INT(test_data: Tuple):
    module, test_data = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        module,
        test_data,
        module.aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_repeat_u85_INT(test_data: Tuple):
    module, test_data = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        module,
        test_data,
        module.aten_op,
        exir_ops=[],
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_repeat_vgf_FP(test_data: Tuple):
    module, args = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        args,
        module.aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_repeat_vgf_INT(test_data: Tuple):
    module, args = test_data()
    pipeline = VgfPipeline[input_t1](
        module,
        args,
        module.aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
