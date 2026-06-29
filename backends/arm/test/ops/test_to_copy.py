# Copyright 2024-2026 Arm Limited and/or its affiliates.
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
input_t2 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


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


class CastAddTensor(torch.nn.Module):
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x.to(dtype=self.target_dtype) + y


class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y


class CastToAddModule(torch.nn.Module):
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype
        self.add = AddModule()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.add(x.to(dtype=self.target_dtype), y)


class CastCatTensor(torch.nn.Module):
    def __init__(self, target_dtype, dim: int):
        super().__init__()
        self.target_dtype = target_dtype
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.cat((x.to(dtype=self.target_dtype), y), dim=self.dim)


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


def test_to_tosa_FP_bf16_requires_extension():
    test_tensor = torch.rand((1, 2, 3, 4), dtype=torch.float32)
    pipeline = OpNotSupportedPipeline[input_t1](
        Cast(torch.bfloat16),
        (test_tensor,),
        {
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1
        },
    )
    pipeline.run()


def test_to_tosa_FP_bf16_with_extension():
    test_tensor = torch.rand((1, 2, 3, 4), dtype=torch.float32)
    pipeline = TosaPipelineFP[input_t1](
        Cast(torch.bfloat16),
        (test_tensor,),
        aten_op=[],
        tosa_extensions=["bf16"],
    )
    pipeline.run()


_TO_COPY_TEST_DATA_FP_FP8 = {
    "fp32_to_fp8e4m3": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float32),
        torch.float8_e4m3fn,
        "fp8e4m3",
    ),
    "fp16_to_fp8e5m2": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float16),
        torch.float8_e5m2,
        "fp8e5m2",
    ),
    "fp8e4m3_to_fp32": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float32).to(torch.float8_e4m3fn),
        torch.float32,
        "fp8e4m3",
    ),
    "fp8e5m2_to_fp16": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float32).to(torch.float8_e5m2),
        torch.float16,
        "fp8e5m2",
    ),
}


def test_to_tosa_FP_fp8e4m3_requires_extension():
    test_tensor = torch.rand((1, 2, 3, 4), dtype=torch.float32)
    pipeline = OpNotSupportedPipeline[input_t1](
        Cast(torch.float8_e4m3fn),
        (test_tensor,),
        {
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1
        },
    )
    pipeline.run()


def test_to_tosa_FP_fp8e5m2_requires_extension():
    test_tensor = torch.rand((1, 2, 3, 4), dtype=torch.float16)
    pipeline = OpNotSupportedPipeline[input_t1](
        Cast(torch.float8_e5m2),
        (test_tensor,),
        {
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1
        },
    )
    pipeline.run()


def test_to_tosa_FP_bf16_to_fp8e4m3_requires_both_extensions():
    test_tensor = torch.rand((1, 2, 3, 4), dtype=torch.bfloat16)
    pipeline = OpNotSupportedPipeline[input_t1](
        Cast(torch.float8_e4m3fn),
        (test_tensor,),
        {
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1
        },
        tosa_extensions=["bf16"],
    )
    pipeline.run()


def test_to_tosa_FP_bf16_to_fp8e5m2_requires_both_extensions():
    test_tensor = torch.rand((1, 2, 3, 4), dtype=torch.bfloat16)
    pipeline = OpNotSupportedPipeline[input_t1](
        Cast(torch.float8_e5m2),
        (test_tensor,),
        {
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1
        },
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", _TO_COPY_TEST_DATA_FP_FP8)
def test_to_tosa_FP_fp8_with_extension(test_data: Tuple):
    test_tensor, new_dtype, tosa_extension = test_data()
    pipeline = TosaPipelineFP[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        tosa_extensions=[tosa_extension],
    )
    pipeline.run()


_TO_COPY_TEST_DATA_BF16_FP8 = {
    "bf16_to_fp8e4m3": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.bfloat16),
        torch.float8_e4m3fn,
        ["bf16", "fp8e4m3"],
    ),
    "fp8e4m3_to_bf16": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float32).to(torch.float8_e4m3fn),
        torch.bfloat16,
        ["bf16", "fp8e4m3"],
    ),
    "bf16_to_fp8e5m2": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.bfloat16),
        torch.float8_e5m2,
        ["bf16", "fp8e5m2"],
    ),
    "fp8e5m2_to_bf16": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float32).to(torch.float8_e5m2),
        torch.bfloat16,
        ["bf16", "fp8e5m2"],
    ),
}


@common.parametrize("test_data", _TO_COPY_TEST_DATA_BF16_FP8)
def test_to_tosa_FP_bf16_fp8_with_extensions(test_data: Tuple):
    test_tensor, new_dtype, tosa_extensions = test_data()
    pipeline = TosaPipelineFP[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        tosa_extensions=tosa_extensions,
    )
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


_TO_COPY_QUANTIZED_IDENTITY_CAST_DATA = {
    "int8_cast_add": lambda: (
        (torch.randn(1, 3, 4, 4) * 10).to(dtype=torch.int8),
        torch.randn(1, 3, 4, 4),
        torch.float32,
    ),
    "int16_cast_add": lambda: (
        (torch.randn(1, 3, 4, 4) * 10).to(dtype=torch.int16),
        torch.randn(1, 3, 4, 4),
        torch.float32,
    ),
    "int32_cast_add": lambda: (
        (torch.randn(1, 3, 4, 4) * 10).to(dtype=torch.int32),
        torch.randn(1, 3, 4, 4),
        torch.float32,
    ),
}


_TO_COPY_QUANTIZED_IDENTITY_CAST_CAT_DATA = {
    "int8_cast_cat": lambda: (
        (torch.randn(1, 2, 4, 4) * 10).to(dtype=torch.int8),
        torch.randn(1, 2, 4, 1),
        torch.float32,
        3,
    ),
    "int16_cast_cat": lambda: (
        (torch.randn(1, 2, 4, 4) * 10).to(dtype=torch.int16),
        torch.randn(1, 2, 4, 1),
        torch.float32,
        3,
    ),
}


@common.parametrize("test_data", _TO_COPY_QUANTIZED_IDENTITY_CAST_DATA)
def test_to_tosa_INT_quantized_identity_cast_add(test_data: Tuple):
    x, y, new_dtype = test_data()
    pipeline = TosaPipelineINT[input_t2](
        CastAddTensor(new_dtype),
        (x, y),
        aten_op=["torch.ops.aten.add.Tensor"],
        exir_op=["executorch_exir_dialects_edge__ops_aten_add_Tensor"],
        qtol=1,
    )
    pipeline.change_args(
        "check_count.exir",
        {
            "torch.ops.higher_order.executorch_call_delegate": 1,
        },
    )
    pipeline.run()


@common.parametrize("test_data", _TO_COPY_QUANTIZED_IDENTITY_CAST_CAT_DATA)
def test_to_tosa_INT_quantized_identity_cast_cat(test_data: Tuple):
    x, y, new_dtype, dim = test_data()
    pipeline = TosaPipelineINT[input_t2](
        CastCatTensor(new_dtype, dim),
        (x, y),
        aten_op=["torch.ops.aten.cat.default"],
        exir_op=["executorch_exir_dialects_edge__ops_aten_cat_default"],
    )
    pipeline.run()


@common.parametrize("test_data", _TO_COPY_QUANTIZED_IDENTITY_CAST_DATA)
def test_to_tosa_INT_quantized_identity_cast_to_unquantized_add_delegated(
    test_data: Tuple,
):
    x, y, new_dtype = test_data()
    pipeline = TosaPipelineINT[input_t2](
        CastToAddModule(new_dtype),
        (x, y),
        aten_op=["torch.ops.aten.add.Tensor"],
        exir_op=["executorch_exir_dialects_edge__ops_aten_add_Tensor"],
    )
    pipeline.quantizer.set_module_name("add", None)
    pipeline.pop_stage("check_not.exir")
    pipeline.change_args(
        "check_count.exir",
        {
            "torch.ops.higher_order.executorch_call_delegate": 1,
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 0,
        },
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

redundant_xfails_FP = {
    "rand_int8_int8": "Tracing graph with quantized input is not supported.",
    "rand_int16_int16": "Tracing graph with quantized input is not supported.",
}

redundant_xfails_INT = redundant_xfails_FP | {
    "rand_fp16_fp16": "FP16 is not supported",
}


@common.parametrize(
    "test_data", _TO_COPY_TEST_DATA_REDUNDANT_CAST, xfails=redundant_xfails_FP
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
    "test_data", _TO_COPY_TEST_DATA_REDUNDANT_CAST, xfails=redundant_xfails_INT
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


_TO_COPY_TEST_DATA_INT_FP = {
    "bool_fp32": lambda: (
        torch.tensor([True, False], dtype=torch.bool),
        torch.float32,
    ),
}


@common.parametrize("test_data", _TO_COPY_TEST_DATA_INT_FP)
@common.SkipIfNoModelConverter
def test_to_vgf_no_quant_bool_fp32(test_data: Tuple):
    test_tensor, new_dtype = test_data()
    pipeline = VgfPipeline[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", _TO_COPY_TEST_DATA_INT_FP)
@common.SkipIfNoModelConverter
def test_to_vgf_quant_bool_fp32(test_data: Tuple):
    test_tensor, new_dtype = test_data()
    pipeline = VgfPipeline[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        quantize=True,
    )
    pipeline.run()
