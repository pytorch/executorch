# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the _to_copy op which is interpreted as a cast for our purposes.
#

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor]  # Input x


class Cast(torch.nn.Module):
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor):
        return x.to(dtype=self.target_dtype)


class CastAdd(torch.nn.Module):
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor):
        return x.to(dtype=self.target_dtype) + x.to(dtype=self.target_dtype)


"""
Tests the _to_copy operation.

Only test unquantized graphs as explicit casting of dtypes messes with the
quantization.
However, the model being exported may have some explicit casting to floating
point dtypes. The casting or their decomposition should be rejected during
partition. This test will be coveraged by class TestToCopy_INT.

Note: This is also covered by test_scalars.py.
"""

_TO_COPY_TEST_DATA_FP = {
    "rand_fp16": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float16), torch.float32),
    "rand_fp32": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float32), torch.float16),
    "rand_int8": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int8),
        torch.float32,
    ),
    "rand_int8_int32": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int8),
        torch.int32,
    ),
    "rand_int32": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.int8,
    ),
}


@common.parametrize("test_data", _TO_COPY_TEST_DATA_FP)
def test_to_tosa_FP(test_data: Tuple):
    test_tensor, new_dtype = test_data()

    pipeline = TosaPipelineFP[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
    )
    # int to int cast is not supported in TOSA+FP profile
    if not new_dtype.is_floating_point and not torch.is_floating_point(test_tensor):
        pipeline.change_args(
            "check_count.exir",
            {
                "torch.ops.higher_order.executorch_call_delegate": 0,
                "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
            },
        )
        pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.parametrize("test_data", _TO_COPY_TEST_DATA_FP)
@common.SkipIfNoModelConverter
def test_to_vgf_no_quant(test_data: Tuple):
    test_tensor, new_dtype = test_data()
    pipeline = VgfPipeline[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        quantize=False,
    )
    pipeline.run()


"""
Casting operations that output floating-point dtypes should be rejected under INT profile,
rather than introducing an invalid dtype into the tosa graph.
For example, x.to(dtype=torch.float32) will be eventually lowered to
exir_ops.edge.dim_order_ops._to_dim_order_copy.default. We should reject this operation
in ToCopySupported::is_node_tosa_supported() before it goes into the delegated graph.
"""
_TO_COPY_TEST_DATA_INT = {
    "rand_int8_fp32": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int8),
        torch.float32,
    ),
    "rand_int16_fp32": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int16),
        torch.float32,
    ),
    "rand_int32_fp32": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.float32,
    ),
    "rand_int32_fp16": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.float16,
    ),
    "rand_int32_bf16": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.bfloat16,
    ),
}


@common.parametrize("test_data", _TO_COPY_TEST_DATA_INT)
def test_to_tosa_INT_not_delegated(test_data: Tuple):
    test_tensor, new_dtype = test_data()

    pipeline = OpNotSupportedPipeline[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        {
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1
        },
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", _TO_COPY_TEST_DATA_INT)
@common.SkipIfNoModelConverter
def test_to_vgf_quant(test_data: Tuple):
    # Op not supported
    pass


_TO_COPY_TEST_DATA_REDUNDANT_CAST = {
    "rand_fp16_fp16": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float16),
        torch.float16,
    ),
    "rand_fp32_fp32": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float32),
        torch.float32,
    ),
    "rand_int8_int8": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int8),
        torch.int8,
    ),
    "rand_int16_int16": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int16),
        torch.int16,
    ),
    "rand_int32_int32": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.int32,
    ),
}

redundant_xfails = {
    "rand_fp16_fp16": "FP16 is not supported",
    "rand_int8_int8": "Tracing graph with quantized input is not supported.",
    "rand_int16_int16": "Tracing graph with quantized input is not supported.",
}


@common.parametrize(
    "test_data", _TO_COPY_TEST_DATA_REDUNDANT_CAST, xfails=redundant_xfails
)
def test_to_tosa_FP_REDUNDANT_CAST(test_data: Tuple):
    test_tensor, new_dtype = test_data()
    pipeline = TosaPipelineFP[input_t1](
        CastAdd(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.parametrize(
    "test_data", _TO_COPY_TEST_DATA_REDUNDANT_CAST, xfails=redundant_xfails
)
def test_to_tosa_INT_REDUNDANT_CAST(test_data: Tuple):
    test_tensor, new_dtype = test_data()
    pipeline = TosaPipelineINT[input_t1](
        CastAdd(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.parametrize("test_data", _TO_COPY_TEST_DATA_REDUNDANT_CAST)
def test_to_tosa_INT_not_delegated_REDUNDANT_CAST(test_data: Tuple):
    test_tensor, new_dtype = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        non_delegated_ops={},  # These are removed outside of the Arm backend so the graph is empty
    )
    pipeline.run()


_TO_COPY_DATA_INT_U55_REJECT = {
    "rand_bool_int8": lambda: (
        torch.randint(0, 2, (1, 2, 3, 4), dtype=torch.bool),
        torch.int8,
    ),
    "rand_int16_bool": lambda: (
        torch.randint(-1000, 1000, (1, 2, 3, 4), dtype=torch.int16),
        torch.bool,
    ),
    "rand_int32_int8": lambda: (
        torch.randint(-1000, 1000, (1, 2, 3, 4), dtype=torch.int32),
        torch.int8,
    ),
}


@common.parametrize("test_data", _TO_COPY_DATA_INT_U55_REJECT)
def test_to_u55_INT(test_data: Tuple):
    test_tensor, new_dtype = test_data()
    pipeline = OpNotSupportedPipeline[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        u55_subset=True,
        quantize=True,
        non_delegated_ops={},  # These are removed outside of the Arm backend so the graph is empty
    )
    pipeline.run()
