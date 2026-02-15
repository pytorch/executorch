# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the max_pool1d operation.

In PyTorch, max_pool1d may be decomposed internally into a sequence of
operations (e.g., unsqueeze -> max_pool2d_with_indices -> getitem -> squeeze),
but this test focuses on ensuring that the max_pool1d aten op is correctly
lowered/quantized and delegated to the expected edge dialect op on the
Arm backend (U55/U85).
"""

from typing import Callable, Tuple

import pytest

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
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


# Test data suite for single-batch tests (N=1), suitable for all targets
test_data_suite = {
    # (test_name, test_data, [kernel_size, stride, padding])
    "simple": lambda: (torch.rand(1, 16, 50), [4, 2, 0]),
    "with_padding": lambda: (torch.rand(1, 16, 50), [3, 2, 1]),
    "stride_1": lambda: (torch.rand(1, 8, 32), [3, 1, 0]),
    "larger_kernel": lambda: (torch.rand(1, 4, 64), [8, 4, 0]),
}

# Multi-batch test data (N>1) - not supported on U55 due to N==1 constraint
test_data_suite_multi_batch = {
    "multi_batch": lambda: (torch.rand(4, 16, 50), [4, 2, 0]),
}

# Combined suite for targets that support multi-batch (TOSA, U85, VGF)
test_data_suite_all = {**test_data_suite, **test_data_suite_multi_batch}

# After PyTorch decomposition, max_pool1d becomes max_pool2d_with_indices
# After to_edge, becomes max_pool2d_with_indices in edge dialect
aten_op = "torch.ops.aten.max_pool1d.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default"


@common.parametrize("test_data", test_data_suite_all)
@pytest.mark.xfail(reason="MaxPool1D not yet supported", strict=False)
def test_max_pool1d_tosa_FP(test_data: Callable):
    """Test max_pool1d with TOSA FP pipeline."""
    test_data, model_params = test_data()
    pipeline = TosaPipelineFP[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_all)
@pytest.mark.xfail(reason="MaxPool1D not yet supported", strict=False)
def test_max_pool1d_tosa_INT(test_data: Callable):
    """Test max_pool1d with TOSA INT pipeline (quantized)."""
    test_data, model_params = test_data()
    pipeline = TosaPipelineINT[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(reason="MaxPool1D not yet supported", strict=False)
def test_max_pool1d_u55_INT(test_data: Callable):
    """Test max_pool1d on Ethos-U55 (quantized)."""
    test_data, model_params = test_data()
    pipeline = EthosU55PipelineINT[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_all)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(reason="MaxPool1D not yet supported", strict=False)
def test_max_pool1d_u85_INT(test_data: Callable):
    """Test max_pool1d on Ethos-U85 (quantized)."""
    test_data, model_params = test_data()
    pipeline = EthosU85PipelineINT[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op,
        exir_ops=[],
    )
    pipeline.run()


# VGF tests
@common.parametrize("test_data", test_data_suite_all)
@common.SkipIfNoModelConverter
def test_max_pool1d_vgf_no_quant(test_data: Callable):
    """Test max_pool1d with VGF pipeline (non-quantized)."""
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_all)
@common.SkipIfNoModelConverter
def test_max_pool1d_vgf_quant(test_data: Callable):
    """Test max_pool1d with VGF pipeline (quantized)."""
    test_data, model_params = test_data()
    pipeline = VgfPipeline[input_t1](
        MaxPool1d(*model_params),
        (test_data,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
