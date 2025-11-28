# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.bitwise_not.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_not_default"

input_t1 = Tuple[torch.Tensor]

test_data_suite = {
    "zeros": torch.zeros(1, 10, 10, 10, dtype=torch.int32),
    "ones": torch.ones(10, 2, 3, dtype=torch.int8),
    "pattern1_int8": 0xAA * torch.ones(1, 2, 2, 2, dtype=torch.int8),
    "pattern1_int16": 0xAAAA * torch.ones(1, 2, 2, 2, dtype=torch.int16),
    "pattern1_int32": 0xAAAAAAAA * torch.ones(1, 2, 2, 2, dtype=torch.int32),
    "pattern2_int8": 0xCC * torch.ones(1, 2, 2, 2, dtype=torch.int8),
    "pattern2_int16": 0xCCCC * torch.ones(1, 2, 2, 2, dtype=torch.int16),
    "pattern2_int32": 0xCCCCCCCC * torch.ones(1, 2, 2, 2, dtype=torch.int32),
    "rand_rank2": torch.randint(-128, 127, (10, 10), dtype=torch.int8),
    "rand_rank4": torch.randint(-128, 127, (1, 10, 10, 10), dtype=torch.int8),
}


class BitwiseNot(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.bitwise_not(x)


@common.parametrize("test_data", test_data_suite)
def test_bitwise_not_tosa_FP(test_data: Tuple):
    # We don't delegate bitwise_not since it is not supported on the FP profile.
    pipeline = OpNotSupportedPipeline[input_t1](
        BitwiseNot(),
        (test_data,),
        {exir_op: 1},
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_bitwise_not_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        BitwiseNot(),
        (test_data,),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_bitwise_not_u55_INT(test_data: Tuple):
    # We don't delegate bitwise_not since it is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t1](
        BitwiseNot(),
        (test_data,),
        {exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_data_suite)
def test_bitwise_not_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        BitwiseNot(),
        (test_data,),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_bitwise_not_vgf_FP(test_data: Tuple):
    # We don't delegate bitwise_not since it is not supported on the FP profile.
    pipeline = OpNotSupportedPipeline[input_t1](
        BitwiseNot(),
        (test_data,),
        {exir_op: 1},
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_bitwise_not_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        BitwiseNot(),
        (test_data,),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
