# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.bitwise_not.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_bitwise_not_default"

input_t1 = Tuple[torch.Tensor]

test_data_suite_non_bool = {
    "zeros": lambda: torch.zeros(1, 10, 10, 10, dtype=torch.int32),
    "ones": lambda: torch.ones(10, 2, 3, dtype=torch.int8),
    "pattern1_int8": lambda: 0xAA * torch.ones(1, 2, 2, 2, dtype=torch.int8),
    "pattern1_int16": lambda: 0xAAAA * torch.ones(1, 2, 2, 2, dtype=torch.int16),
    "pattern1_int32": lambda: 0xAAAAAAAA * torch.ones(1, 2, 2, 2, dtype=torch.int32),
    "pattern2_int8": lambda: 0xCC * torch.ones(1, 2, 2, 2, dtype=torch.int8),
    "pattern2_int16": lambda: 0xCCCC * torch.ones(1, 2, 2, 2, dtype=torch.int16),
    "pattern2_int32": lambda: 0xCCCCCCCC * torch.ones(1, 2, 2, 2, dtype=torch.int32),
    "rand_rank2": lambda: torch.randint(-128, 127, (10, 10), dtype=torch.int8),
    "rand_rank4": lambda: torch.randint(-128, 127, (1, 10, 10, 10), dtype=torch.int8),
}

test_data_suite_bool = {
    "pattern_bool": lambda: torch.tensor([True, False, True], dtype=torch.bool),
}

test_data_suite = {**test_data_suite_non_bool, **test_data_suite_bool}


class BitwiseNot(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.bitwise_not(x)


@common.parametrize("test_data", test_data_suite_non_bool)
def test_bitwise_not_tosa_FP(test_data: Tuple):
    # We don't delegate bitwise_not since it is not supported on the FP profile.
    pipeline = OpNotSupportedPipeline[input_t1](
        BitwiseNot(),
        (test_data(),),
        {exir_op: 1},
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_bool)
def test_bitwise_not_tosa_FP_bool(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t1](
        BitwiseNot(),
        (test_data(),),
        aten_op,
        "executorch_exir_dialects_edge__ops_aten_logical_not_default",
        atol=0,
        rtol=0,
        qtol=0,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_bitwise_not_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t1](
        BitwiseNot(),
        (test_data(),),
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_bitwise_not_u55_INT(test_data: Tuple):
    # We don't delegate bitwise_not since it is not supported on U55.
    pipeline = OpNotSupportedPipeline[input_t1](
        BitwiseNot(),
        (test_data(),),
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
        (test_data(),),
        aten_ops=aten_op,
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_bitwise_not_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        BitwiseNot(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_bitwise_not_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        BitwiseNot(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
