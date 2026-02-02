# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for max_pool1d operation.

max_pool1d is decomposed by DecomposeMaxPool1dPass into:
    view_copy -> max_pool2d -> view_copy

This is done before quantization to ensure proper qparams propagation.
The test verifies that the decomposed pattern is correctly quantized and
delegated to the Arm backend (U55/U85).
"""

from typing import Tuple

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t1 = Tuple[torch.Tensor]


class MaxPool1d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.max_pool_1d = torch.nn.MaxPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.max_pool_1d(x)


# Test data for TOSA pipelines (no stride constraints)
test_data_suite = {
    # (test_name, test_data, [kernel_size, stride, padding])
    "simple": lambda: (torch.rand(1, 16, 50), [4, 2, 0]),
    "with_padding": lambda: (torch.rand(1, 16, 50), [3, 2, 1]),
    "stride_1": lambda: (torch.rand(1, 8, 32), [3, 1, 0]),
    "larger_kernel": lambda: (torch.rand(1, 4, 64), [8, 4, 0]),
    "multi_batch": lambda: (torch.rand(4, 16, 50), [4, 2, 0]),
}

# Test data for U55/U85 pipelines (stride must be <= 3)
test_data_suite_u55 = {
    # (test_name, test_data, [kernel_size, stride, padding])
    "simple": lambda: (torch.rand(1, 16, 50), [4, 2, 0]),
    "with_padding": lambda: (torch.rand(1, 16, 50), [3, 2, 1]),
    "stride_1": lambda: (torch.rand(1, 8, 32), [3, 1, 0]),
    "stride_3": lambda: (torch.rand(1, 4, 64), [8, 3, 0]),
}

# max_pool1d is decomposed before quantization by DecomposeMaxPool1dPass
# After the pass, max_pool1d becomes view_copy -> max_pool2d -> view_copy
# So for the INT (quantized) tests we should not expect max_pool1d
aten_op_INT = "torch.ops.aten.view_copy.default"
# For FP (non-quantized) tests, max_pool1d remains
aten_op_FP = "torch.ops.aten.max_pool1d.default"
# After decomposition and passes, becomes max_pool2d in edge dialect
exir_op = "executorch_exir_dialects_edge__ops_aten_max_pool2d_default"


@common.parametrize("test_data", test_data_suite)
def test_max_pool1d_tosa_FP(test_data: torch.Tensor):
    """Test max_pool1d with TOSA FP pipeline."""
    test_data, model_params = test_data()
    pipeline = TosaPipelineFP[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op_FP,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_max_pool1d_tosa_INT(test_data: torch.Tensor):
    """Test max_pool1d with TOSA INT pipeline (quantized)."""
    test_data, model_params = test_data()
    pipeline = TosaPipelineINT[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op_INT,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_u55)
@common.XfailIfNoCorstone300
def test_max_pool1d_u55_INT(test_data: torch.Tensor):
    """Test max_pool1d on Ethos-U55 (quantized).

    Note: U55 has stride constraint <= 3, so we use test_data_suite_u55
    which excludes larger_kernel (stride=4).
    """
    test_data, model_params = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op_INT,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_u55)
@common.XfailIfNoCorstone320
def test_max_pool1d_u85_INT(test_data: torch.Tensor):
    """Test max_pool1d on Ethos-U85 (quantized).

    Note: U85 has stride constraint <= 3, so we use test_data_suite_u55
    which excludes larger_kernel (stride=4).
    """
    test_data, model_params = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op_INT,
        exir_ops=[],
    )
    pipeline.run()
