# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.test import common, conftest

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    OpNotSupportedPipeline,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.avg_pool2d.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"

input_t = Tuple[torch.Tensor]


class AvgPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        padding: int | Tuple[int, int],
    ):
        super().__init__()
        self.avg_pool_2d = torch.nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        return self.avg_pool_2d(x)


test_modules = {
    "zeros": lambda: (AvgPool2d(4, 2, 0), (torch.zeros(1, 16, 50, 32),)),
    "ones": lambda: (AvgPool2d(4, 2, 0), (torch.ones(1, 16, 50, 32),)),
    "rand": lambda: (AvgPool2d(4, 2, 0), (torch.rand(1, 16, 50, 32),)),
    "randn": lambda: (AvgPool2d(4, 2, 0), (torch.randn(1, 16, 50, 32),)),
    "kernel_3x3_stride_1_pad_1": lambda: (
        AvgPool2d((3, 3), (1, 1), 1),
        (torch.rand(1, 16, 50, 32),),
    ),
    "kernel_3x2_stride_1x2_pad_1x0": lambda: (
        AvgPool2d((3, 2), (1, 2), (1, 0)),
        (torch.rand(1, 16, 50, 32),),
    ),
    "kernel_4x6_stride_1x2_pad_2x3": lambda: (
        AvgPool2d((4, 6), (1, 2), (2, 3)),
        (torch.rand(1, 16, 50, 32),),
    ),
    "non_divisible_window": lambda: (
        AvgPool2d(3, 2, 1),
        (torch.rand(1, 16, 112, 112),),
    ),
    "non_divisible_window_height": lambda: (
        AvgPool2d(3, (2, 1), 1),
        (torch.rand(1, 16, 56, 56),),
    ),
    "non_divisible_window_width": lambda: (
        AvgPool2d(3, (1, 2), 1),
        (torch.rand(1, 16, 56, 56),),
    ),
}


@common.parametrize("test_module", test_modules)
def test_avg_pool2d_tosa_MI(test_module):
    model, input_tensor = test_module()

    pipeline = TosaPipelineMI[input_t](
        model,
        input_tensor,
        aten_op,
        exir_op,
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    if conftest.is_option_enabled("tosa_ref_model"):
        pipeline.change_args("run_method_and_compare_outputs", qtol=1, atol=1, rtol=1)
        pipeline.run()


@common.parametrize("test_module", test_modules)
def test_avg_pool2d_tosa_BI(test_module):
    model, input_tensor = test_module()

    pipeline = TosaPipelineBI[input_t](
        model,
        input_tensor,
        aten_op,
        exir_op,
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    if conftest.is_option_enabled("tosa_ref_model"):
        pipeline.change_args("run_method_and_compare_outputs", qtol=1, atol=1, rtol=1)
        pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone300
def test_avg_pool2d_u55_BI(test_module):
    model, input_tensor = test_module()

    pipeline = EthosU55PipelineBI[input_t](
        model,
        input_tensor,
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1, atol=1, rtol=1)
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone320
def test_avg_pool2d_u85_BI(test_module):
    model, input_tensor = test_module()

    pipeline = EthosU85PipelineBI[input_t](
        model,
        input_tensor,
        aten_op,
        exir_op,
        run_on_fvp=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1, atol=1, rtol=1)

    pipeline.run()


reject_modules = {
    "kernel_1x1_stride_1_pad_0": lambda: (AvgPool2d(1, 1, 0), torch.rand(2, 5, 5, 5)),
    "kernel_2x9_stride_1_pad_1": lambda: (
        AvgPool2d((2, 9), 1, 1),
        torch.rand(1, 16, 5, 32),
    ),
    "kernel_1x4_stride_0_pad_0": lambda: (
        AvgPool2d(1, 4, 0),
        torch.rand(1, 10, 10, 10),
    ),
    "kernel_1x257_stride_1_pad_0_large": lambda: (
        AvgPool2d((1, 257), 1, 0),
        torch.rand(1, 16, 5, 300),
    ),
    "kernel_800x90_stride_1_pad_0_extreme": lambda: (
        AvgPool2d((800, 90), 1, 0),
        torch.rand(1, 16, 850, 100),
    ),
}


@common.parametrize("reject_module", reject_modules)
def test_avg_pool2d_u55_BI_not_delegated(reject_module):

    model, test_data = reject_module()

    pipeline = OpNotSupportedPipeline[input_t](
        module=model,
        test_data=(test_data,),
        non_delegated_ops={},
        n_expected_delegates=0,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()
