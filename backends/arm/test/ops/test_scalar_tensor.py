# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
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

float_test_data_suite = {
    "scalar_tensor_float_1": lambda: (3.7, torch.float32, torch.rand((1, 2, 3, 4))),
    "scalar_tensor_float_2": lambda: (66, torch.float32, torch.rand((1, 2, 3))),
}

int_test_data_suite = {
    "scalar_tensor_int32": lambda: (
        33,
        torch.int32,
        torch.randint(0, 10, (1, 2), dtype=torch.int32),
    ),
    "scalar_tensor_int8": lambda: (
        8,
        torch.int8,
        torch.rand(1, 2, 3),
    ),
    "scalar_tensor_int16": lambda: (
        16 * 16 * 16,
        torch.int16,
        torch.rand((1,)).unsqueeze(0),  # Rank 0 inputs not supported
    ),
}


class ScalarTensor(torch.nn.Module):
    aten_op = "torch.ops.aten.scalar_tensor.default"

    def __init__(self, scalar, dtype=torch.float32):
        super().__init__()
        self.scalar = scalar
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        return torch.scalar_tensor(self.scalar, dtype=self.dtype) + x


@common.parametrize(
    "test_data",
    int_test_data_suite | float_test_data_suite,
)
def test_scalar_tensor_tosa_FP(test_data):  # Note TOSA FP supports all types
    scalar, dtype, data = test_data()
    TosaPipelineFP(
        ScalarTensor(scalar, dtype),
        tuple(data),
        ScalarTensor.aten_op,
    ).run()


@common.parametrize(
    "test_data",
    int_test_data_suite | float_test_data_suite,
)
def test_scalar_tensor_tosa_INT(test_data):
    scalar, dtype, data = test_data()
    pipeline: TosaPipelineINT = TosaPipelineINT(
        ScalarTensor(scalar, dtype),
        tuple(data),
        ScalarTensor.aten_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", float_test_data_suite)
@common.XfailIfNoCorstone300
def test_scalar_tensor_u55_INT(test_data):
    scalar, dtype, data = test_data()
    EthosU55PipelineINT(
        ScalarTensor(scalar, dtype),
        tuple(data),
        ScalarTensor.aten_op,
    ).run()


@common.parametrize("test_data", float_test_data_suite)
@common.XfailIfNoCorstone320
def test_scalar_tensor_u85_INT(test_data):
    scalar, dtype, data = test_data()
    EthosU85PipelineINT(
        ScalarTensor(scalar, dtype),
        tuple(data),
        ScalarTensor.aten_op,
    ).run()


@common.parametrize("test_data", float_test_data_suite)
@common.SkipIfNoModelConverter
@pytest.mark.xfail(reason="MLETORCH-1410: Tensor dimension count not supported: 0")
def test_scalar_tensor_vgf_FP(test_data):
    scalar, dtype, data = test_data()
    pipeline = VgfPipeline(
        ScalarTensor(scalar, dtype),
        tuple(data),
        ScalarTensor.aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", int_test_data_suite)
@common.SkipIfNoModelConverter
@pytest.mark.xfail(reason="MLETORCH-1410: Tensor dimension count not supported: 0")
def test_scalar_tensor_vgf_INT(test_data):
    scalar, dtype, data = test_data()
    pipeline = VgfPipeline(
        ScalarTensor(scalar, dtype),
        tuple(data),
        ScalarTensor.aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
