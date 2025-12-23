# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

test_data_suite = {
    # (test_name, test_data)
    "ones_two_tensors": lambda: ((torch.ones(1), torch.ones(1)), 0),
    "ones_and_rand_three_tensors": lambda: (
        (torch.ones(1, 2), torch.randn(1, 2), torch.randn(1, 2)),
        1,
    ),
    "ones_and_rand_four_tensors": lambda: (
        (
            torch.ones(1, 2, 5),
            torch.randn(1, 2, 5),
            torch.randn(1, 2, 5),
            torch.randn(1, 2, 5),
        ),
        -1,
    ),
    "rand_two_tensors": lambda: (
        (torch.randn(2, 2, 4), torch.randn(2, 2, 4)),
        2,
    ),
    "rand_two_tensors_dim_0": lambda: (
        (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
    ),
    "rand_two_tensors_dim_2": lambda: (
        (torch.randn(2, 2, 3, 5), torch.randn(2, 2, 3, 5)),
        2,
    ),
    "rand_large": lambda: (
        (
            10000 * torch.randn(2, 3, 1, 4),
            torch.randn(2, 3, 1, 4),
            torch.randn(2, 3, 1, 4),
        ),
        -3,
    ),
}


class Stack(nn.Module):
    aten_op = "torch.ops.aten.stack.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_cat_default"

    def forward(self, n: tuple[torch.Tensor, ...], dim: int = 0):
        return torch.stack(n, dim)


input_t1 = Tuple[torch.Tensor]


@common.parametrize("test_module", test_data_suite)
def test_stack_tosa_FP(test_module: input_t1):
    test_data = test_module()
    pipeline = TosaPipelineFP[input_t1](
        Stack(),
        test_data,
        aten_op=Stack.aten_op,
        exir_op=Stack.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.parametrize("test_module", test_data_suite)
def test_stack_tosa_INT(test_module: input_t1):
    test_data = test_module()
    pipeline = TosaPipelineINT[input_t1](
        Stack(),
        test_data,
        aten_op=Stack.aten_op,
        exir_op=Stack.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_module", test_data_suite)
def test_stack_u55_INT(test_module: input_t1):
    test_data = test_module()
    pipeline = EthosU55PipelineINT[input_t1](
        Stack(),
        test_data,
        aten_ops=Stack.aten_op,
        exir_ops=Stack.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_module", test_data_suite)
def test_stack_u85_INT(test_module: input_t1):
    test_data = test_module()
    pipeline = EthosU85PipelineINT[input_t1](
        Stack(),
        test_data,
        aten_ops=Stack.aten_op,
        exir_ops=Stack.exir_op,
        use_to_edge_transform_and_lower=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_suite)
def test_stack_vgf_no_quant(test_module: input_t1):
    test_data = test_module()
    pipeline = VgfPipeline[input_t1](
        Stack(),
        test_data,
        aten_op=Stack.aten_op,
        exir_op=Stack.exir_op,
        use_to_edge_transform_and_lower=False,
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_module", test_data_suite)
def test_stack_vgf_quant(test_module: input_t1):
    test_data = test_module()
    pipeline = VgfPipeline[input_t1](
        Stack(),
        test_data,
        aten_op=Stack.aten_op,
        exir_op=Stack.exir_op,
        use_to_edge_transform_and_lower=False,
        quantize=True,
    )
    pipeline.run()
