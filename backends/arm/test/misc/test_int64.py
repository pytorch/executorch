# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester


class ConstAdd(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, bias=0):
        super().__init__()
        self.dtype = dtype
        self.bias = bias

    def forward(self, x: torch.Tensor):
        c = torch.arange(self.bias, self.bias + 10, 1, dtype=self.dtype)
        # Add explicit float cast to make quantization work, will be inserted by type promotion otherwise.
        return x + c.to(torch.float32)


class BufferAdd(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, bias=0):
        super().__init__()
        self.dtype = dtype
        self.buffer = torch.arange(0, 10, 1, dtype=self.dtype) + bias
        self.bias = bias

    def forward(self, x: torch.Tensor):
        c = self.buffer
        # Add explicit float cast to make quantization work, will be inserted by type promotion otherwise.
        return x + c.to(torch.float32)


class ConstChainAdd(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        c = torch.arange(0, 10, 1, dtype=self.dtype).reshape((2, 5)).unsqueeze(-1)
        # Add explicit float cast to make quantization work, will be inserted by type promotion otherwise.
        return x + c.to(torch.float32)


class BufferChainAdd(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.buffer = torch.arange(0, 10, 1, dtype=self.dtype)

    def forward(self, x: torch.Tensor):
        c = self.buffer.reshape((2, 5)).unsqueeze(-1)
        # Add explicit float cast to make quantization work, will be inserted by type promotion otherwise.
        return x + c.to(torch.float32)


test_data_suite = {
    "fp32_in+int64_buffer": (BufferAdd(torch.int64), (torch.rand(10) - 0.5,)),
    "fp32_in+int64_buffer_overflow": (
        BufferAdd(torch.int64, 2**40),
        (torch.rand(10) - 0.5,),
    ),
    "fp32_in+int64_const": (ConstAdd(torch.int64), (torch.rand(10) - 0.5,)),
    "fp32_in+int64_const_overflow": (
        ConstAdd(torch.int64, 2**40),
        (torch.rand(10) - 0.5,),
    ),
    "fp32_in+int64_buffer_chain": (
        BufferChainAdd(torch.int64),
        (torch.rand(2, 5, 3) - 0.5,),
    ),
    "fp32_in+int64_const_chain": (
        ConstChainAdd(torch.int64),
        (torch.rand(2, 5, 3) - 0.5,),
    ),
    "int64_in+float_const_chain": (
        ConstChainAdd(torch.float32),
        (torch.randint(0, 10, (2, 5, 3)),),
    ),
}


@common.parametrize("test_data", test_data_suite)
def test_int64_tosa_FP(test_data: Tuple):
    model, inputs = test_data
    (
        ArmTester(
            model,
            inputs,
            common.get_tosa_compile_spec("TOSA-1.0+FP"),
        )
        .export()
        .to_edge_transform_and_lower()
        .to_executorch()
        .run_method_and_compare_outputs(inputs)
    )


@common.parametrize("test_data", test_data_suite)
def test_int64_tosa_INT(test_data: Tuple):
    model, inputs = test_data
    (
        ArmTester(model, inputs, common.get_tosa_compile_spec("TOSA-1.0+INT"))
        .quantize()
        .export()
        .to_edge_transform_and_lower()
        .to_executorch()
        .run_method_and_compare_outputs(inputs)
    )


class Int64Op(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, *args):
        return self.op(*args)


_WIDE = torch.tensor([2**40, -(2**38), 2**31, -1], dtype=torch.int64)
_WIDE_2 = torch.tensor([2**39, 2**37, 1, -(2**40)], dtype=torch.int64)
_NARROW = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
_BYTES = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
_COND = torch.tensor([True, False, True, False])
_FLAGS = torch.tensor([False, True, True, False])


class Int64Buffer(torch.nn.Module):
    def __init__(self, op, *buffers):
        super().__init__()
        self.op = op
        for i, buffer in enumerate(buffers or (_WIDE,)):
            self.register_buffer(f"buffer_{i}", buffer)

    def forward(self, *args):
        return self.op(*self.buffers(), *args)


class Int64Constant(torch.nn.Module):
    def __init__(self, op, value):
        super().__init__()
        self.op = op
        self.value = value

    def forward(self, *args):
        return self.op(self.value, *args)


# A listed operator delegates when every operand casts to int64; an unlisted
# operator or an operand narrower than int32 falls back instead. to_edge widens
# the operands of some operators, such as cat, before partitioning. The widening
# cast delegates from int32 or bool and stays on the CPU from int8.
int64_extension_partition_suite = {
    "add": (lambda x, y: x + y, (_WIDE, _WIDE_2), 1),
    "pad": (lambda x: torch.nn.functional.pad(x, (1, 1)), (_WIDE,), 1),
    "cumsum": (lambda x: torch.cumsum(x, 0), (_WIDE,), 0),
    "widen_narrow": (lambda x: x.to(torch.int64), (_NARROW,), 1),
    "widen_bool": (lambda x: x.to(torch.int64), (_FLAGS,), 1),
    "cat_narrow_promoted": (lambda x, y: torch.cat([x, y]), (_WIDE, _NARROW), 1),
    "or_narrow": (torch.bitwise_or, (_WIDE, _NARROW), 1),
    "cat_byte_promoted": (lambda x, y: torch.cat([x, y]), (_WIDE, _BYTES), 1),
    "or_byte": (torch.bitwise_or, (_WIDE, _BYTES), 0),
    "eq_bool": (lambda x, y: x == y, (_WIDE, _FLAGS), 1),
    "cat_bool_promoted": (lambda x, y: torch.cat([x, y]), (_FLAGS, _WIDE), 1),
    "or_bool": (torch.bitwise_or, (_FLAGS, _WIDE), 1),
    "where": (torch.where, (_COND, _WIDE, _WIDE_2), 1),
    "where_buffer": (
        Int64Buffer(lambda wide, cond, x: torch.where(cond, x, wide)),
        (_COND, _WIDE_2),
        1,
    ),
    "gt_buffer_narrow": (Int64Buffer(lambda wide, x: x > wide), (_NARROW,), 1),
    "gt_small_buffer_byte": (
        Int64Buffer(lambda small, x: x > small, torch.tensor([10, 20, 1, 2])),
        (_BYTES,),
        1,
    ),
    "index_buffers": (
        Int64Buffer(
            lambda rows, columns, x: x[rows, columns],
            torch.tensor([0, 1]),
            torch.tensor([2, 0]),
        ),
        (torch.arange(6, dtype=torch.int32).reshape(2, 3),),
        1,
    ),
    "gt_scalar_buffer_narrowed_buffer": (
        Int64Buffer(
            lambda wide, small, x: x
            + (wide > small).to(torch.int32)
            + small.to(torch.int32),
            torch.tensor(2**40),
            torch.tensor([1]),
        ),
        (torch.tensor([0], dtype=torch.int32),),
        1,
    ),
    "where_buffer_gt_byte": (
        Int64Buffer(
            lambda wide, x, y: torch.where(x > wide, wide, y).abs(),
            torch.tensor([-(2**31)]),
        ),
        (torch.tensor([0], dtype=torch.int8), torch.tensor([0], dtype=torch.int32)),
        1,
    ),
    "index_constant_where": (
        Int64Constant(
            lambda index, cond, x: (
                torch.arange(4, dtype=torch.int32)[index],
                torch.where(cond, index, x),
            ),
            torch.tensor([3, 1]),
        ),
        (_COND[:2], _NARROW[:2]),
        1,
    ),
    "gt_scalar_constant": (
        Int64Constant(lambda limit, x: x > limit, torch.tensor(2**40)),
        (torch.tensor([1], dtype=torch.int32),),
        1,
    ),
    "where_buffer_gt_scalar": (
        Int64Buffer(lambda wide, x, y: torch.where(wide > 7, x, y)),
        (_WIDE, _WIDE_2),
        1,
    ),
    "where_scalar_tensor": (torch.where, (_COND, _NARROW, torch.tensor(2**40)), 0),
    "where_scalar_buffer": (
        Int64Buffer(
            lambda value, cond, x: torch.where(cond, x, value),
            torch.tensor(2**40 + 7),
        ),
        (_COND, _NARROW),
        0,
    ),
    "lt_float": (lambda x: x < 0.5, (torch.tensor([-1, 0, 1]),), 0),
    "strided_slice": (lambda x: x[::2], (torch.arange(5) * 2**33,), 1),
    "gt": (lambda x, y: x > y, (_WIDE, _WIDE_2), 1),
    "gt_scalar": (lambda x: x > 7, (_WIDE,), 1),
    "full_like": (lambda x: torch.full_like(x, 5) + x, (_WIDE,), 1),
    "zeros_like": (lambda x: torch.zeros_like(x) + x, (_WIDE,), 1),
    "add_const": (lambda x: x + 2**40, (_WIDE,), 1),
    "full_like_output": (lambda x: (x > 0, torch.full_like(x, 2**40)), (_WIDE,), 1),
    "full_like_dtype": (
        lambda x: torch.full_like(x, 2**40, dtype=torch.int64) + x,
        (_NARROW,),
        1,
    ),
}


@common.parametrize("test_data", int64_extension_partition_suite)
def test_int64_extension_partition_tosa_INT(test_data: Tuple):
    op, inputs, delegates = test_data
    tester = (
        ArmTester(
            op if isinstance(op, torch.nn.Module) else Int64Op(op),
            inputs,
            common.get_tosa_compile_spec("TOSA-1.1+INT+int64"),
        )
        .export()
        .to_edge_transform_and_lower()
        .check_count({"torch.ops.higher_order.executorch_call_delegate": delegates})
    )
    if delegates:
        tester.to_executorch().run_method_and_compare_outputs(inputs)


@common.parametrize(
    "test_data",
    {"narrow": (_NARROW, 0), "bool": (_FLAGS, 0), "byte": (_BYTES, 1)},
)
def test_int64_extension_widening_cast_placement_tosa_INT(test_data: Tuple):
    narrow, cpu_casts = test_data
    (
        ArmTester(
            Int64Op(lambda x, y: torch.cat([x, y])),
            (_WIDE, narrow),
            common.get_tosa_compile_spec("TOSA-1.1+INT+int64"),
        )
        .export()
        .to_edge_transform_and_lower()
        .check_count(
            {
                "torch.ops.higher_order.executorch_call_delegate": 1,
                "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": cpu_casts,
            }
        )
    )
