# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
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

input_t = tuple[torch.Tensor]
test_data_t = tuple[int, torch.dtype]


class EyeAdd(torch.nn.Module):
    aten_op: str = "torch.ops.aten.eye.default"

    def __init__(self, n: int, dtype: torch.dtype):
        super().__init__()
        self.args = (n,)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(*self.args, dtype=self.dtype) + x

    test_data: dict[str, test_data_t] = {
        "10x3x3": (lambda: (torch.randn(10, 3, 3),), (3, torch.float32)),
        "10x1": (lambda: (torch.randn(10, 1),), (10, torch.float32)),
        "int32_int32": (
            lambda: (torch.randint(0, 10, [10], dtype=torch.int32),),
            (10, torch.int32),
        ),
    }

    # Mixed dtypes - the eye op is delegated, but it leads to a non-delegated add op.
    test_data_mixed_dtypes: dict[str, test_data_t] = {
        "fp32_int64": (lambda: (torch.randn(10),), (10, torch.int64)),
        "fp32_int32": (lambda: (torch.randn(10),), (10, torch.int32)),
    }


# skip test since int32 isn't support on FP profile
# "int32_int32": "view/RESHAPE of integer tensor is not supported for +FP profile"
@pytest.mark.skip(reason="MLETORCH-1274 Improve data type checks during partitioning")
@common.parametrize("test_data", EyeAdd.test_data)
def test_eye_tosa_FP(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t](
        EyeAdd(*init_data),
        input_data(),
        EyeAdd.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", EyeAdd.test_data | EyeAdd.test_data_mixed_dtypes)
def test_eye_tosa_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineINT[input_t](
        EyeAdd(*init_data),
        input_data(),
        EyeAdd.aten_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", EyeAdd.test_data)
@common.XfailIfNoCorstone300
def test_eye_u55_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU55PipelineINT[input_t](
        EyeAdd(*init_data),
        input_data(),
        EyeAdd.aten_op,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", EyeAdd.test_data)
@common.XfailIfNoCorstone320
def test_eye_u85_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU85PipelineINT[input_t](
        EyeAdd(*init_data),
        input_data(),
        EyeAdd.aten_op,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


# skip since int32 isn't support on FP profile
# "int32_int32": "view/RESHAPE of integer tensor is not supported for +FP profile"
@pytest.mark.skip(reason="MLETORCH-1274 Improve data type checks during partitioning")
@common.parametrize(
    "test_data",
    EyeAdd.test_data,
)
@common.SkipIfNoModelConverter
def test_eye_vgf_FP(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = VgfPipeline[input_t](
        EyeAdd(*init_data),
        input_data(),
        EyeAdd.aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    EyeAdd.test_data,
)
@common.SkipIfNoModelConverter
def test_eye_vgf_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = VgfPipeline[input_t](
        EyeAdd(*init_data),
        input_data(),
        EyeAdd.aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize(
    "test_data",
    EyeAdd.test_data_mixed_dtypes,
)
def test_eye_tosa_INT_not_delegated(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = OpNotSupportedPipeline[input_t](
        EyeAdd(*init_data),
        input_data(),
        non_delegated_ops={"executorch_exir_dialects_edge__ops_aten_add_Tensor": 1},
        n_expected_delegates=1,
        quantize=True,
    )
    pipeline.run()
