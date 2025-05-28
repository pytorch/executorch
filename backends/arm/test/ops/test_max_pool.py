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


test_data_suite = {
    # (test_name, test_data, [kernel_size, stride, padding])
    "zeros": lambda: (torch.zeros(1, 1, 4, 8), [2, 2, 1]),
    "ones": lambda: (torch.ones(1, 16, 50, 32), [4, 2, 0]),
    "rand": lambda: (torch.rand(1, 16, 52, 16), [4, 3, 0]),
    "non_divisible": lambda: (torch.rand(1, 16, 112, 112), [3, 2, 1]),
    "non_divisible_window_height": lambda: (torch.rand(1, 16, 56, 56), [3, (2, 1), 1]),
    "non_divisible_window_width": lambda: (torch.rand(1, 16, 56, 56), [3, (1, 2), 1]),
}

test_data_suite_mult_batches = {
    "randn": lambda: (torch.randn(5, 16, 50, 32), [4, 2, 0]),
}


aten_op = "torch.ops.aten.max_pool2d.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_max_pool2d_default"

input_t1 = Tuple[torch.Tensor]


class MaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        padding: int | Tuple[int, int],
    ):
        super().__init__()
        self.max_pool_2d = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        return self.max_pool_2d(x)


@common.parametrize("test_data", test_data_suite)
def test_max_pool2d_tosa_MI(test_data: torch.Tensor):
    test_data, model_params = test_data()
    pipeline = TosaPipelineMI[input_t1](
        MaxPool2d(*model_params), (test_data,), aten_op, exir_op
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_max_pool2d_tosa_BI(test_data: torch.Tensor):
    test_data, model_params = test_data()
    pipeline = TosaPipelineBI[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_max_pool2d_u55_BI(test_data: torch.Tensor):
    test_data, model_params = test_data()
    EthosU55PipelineBI[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_max_pool2d_u85_BI(test_data: torch.Tensor):
    test_data, model_params = test_data()
    EthosU85PipelineBI[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", test_data_suite_mult_batches)
def test_max_pool2d_tosa_MI_mult_batches(test_data: torch.Tensor):
    test_data, model_params = test_data()
    pipeline = TosaPipelineMI[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_mult_batches)
def test_max_pool2d_tosa_BI_mult_batches(test_data: torch.Tensor):
    test_data, model_params = test_data()
    pipeline = TosaPipelineBI[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


x_fail = {"randn": "MLETORCH-986: Numerical issues with mutli batches."}


@common.parametrize("test_data", test_data_suite_mult_batches, x_fail)
@common.XfailIfNoCorstone300
def test_max_pool2d_u55_BI_mult_batches(test_data: torch.Tensor):
    test_data, model_params = test_data()
    EthosU55PipelineBI[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    ).run()


@common.parametrize("test_data", test_data_suite_mult_batches, x_fail)
@common.XfailIfNoCorstone320
def test_max_pool2d_u85_BI_mult_batches(test_data: torch.Tensor):
    test_data, model_params = test_data()
    EthosU85PipelineBI[input_t1](
        MaxPool2d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        run_on_fvp=True,
        use_to_edge_transform_and_lower=True,
    ).run()


reject_data_suite = {
    "reject_1": lambda: (MaxPool2d(1, 4, 0), torch.rand(1, 10, 10, 10)),
    "reject_2": lambda: (MaxPool2d((1, 257), 1, 0), torch.rand(1, 16, 5, 300)),
    "reject_3": lambda: (MaxPool2d((800, 90), 1, 0), torch.rand(1, 16, 850, 100)),
}


@common.parametrize("test_data", reject_data_suite)
@common.XfailIfNoCorstone300
def test_max_pool2d_u55_BI_failure_set(test_data: Tuple):
    module, test_data = test_data()
    pipeline = EthosU55PipelineBI[input_t1](
        module,
        (test_data,),
        aten_op,
        exir_op,
        run_on_fvp=False,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()
