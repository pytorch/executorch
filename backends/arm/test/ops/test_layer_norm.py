# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)


class LayerNorm(torch.nn.Module):

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        has_bias: bool = True,
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=has_bias,
        )
        if elementwise_affine:
            self.layer_norm.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        if has_bias:
            self.layer_norm.bias = torch.nn.Parameter(torch.rand(normalized_shape))

    def forward(self, x):
        return self.layer_norm(x)


input_t = tuple[torch.Tensor]
test_data_suite = {
    "randn_last_dim": ((torch.randn(1, 5, 5, 5),), LayerNorm([5])),
    "rand_last_two_dims": ((torch.rand(1, 5, 5, 5),), LayerNorm([5, 5])),
    "rand_last_two_dims_not_elementwise_affine": (
        (torch.rand(1, 5, 5, 5),),
        LayerNorm([5, 5], 1e-5, False),
    ),
    "rand_last_two_dims_not_elementwise_affine_no_bias": (
        (torch.rand(1, 5, 5, 5),),
        LayerNorm([5, 5], 1e-5, False, False),
    ),
    "randn_last_three_dims": ((torch.randn(1, 15, 10, 5),), LayerNorm([15, 10, 5])),
    "randn_last_three_dims_no_bias": (
        (torch.randn(1, 15, 10, 5),),
        LayerNorm([15, 10, 5], 1e-2, False, False),
    ),
}


@common.parametrize("test_data", test_data_suite)
def test_native_layer_norm_tosa_MI(test_data):
    pipeline = TosaPipelineMI[input_t](
        test_data[1],
        test_data[0],
        "torch.ops.aten.layer_norm.default",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_native_layer_norm_tosa_BI(test_data):
    pipeline = TosaPipelineBI[input_t](
        test_data[1],
        test_data[0],
        "torch.ops.aten.sub.Tensor",  # Just check for sub op included in the layernorm decomposition
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_native_layer_norm_u55_BI(test_data):
    pipeline = EthosU55PipelineBI[input_t](
        test_data[1],
        test_data[0],
        "torch.ops.aten.sub.Tensor",  # Just check for sub op included in the layernorm decomposition
        run_on_fvp=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_native_layer_norm_u85_BI(test_data):
    pipeline = EthosU85PipelineBI[input_t](
        test_data[1],
        test_data[0],
        "torch.ops.aten.sub.Tensor",  # Just check for sub op included in the layernorm decomposition
        run_on_fvp=True,
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()
