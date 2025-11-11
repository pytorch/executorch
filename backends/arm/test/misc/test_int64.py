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
    "int64_in+float_const": (
        ConstAdd(torch.float32),
        (torch.randint(0, 10, (10,)),),
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
            common.get_tosa_compile_spec("TOSA-1.0+FP", custom_path="tosa/int64"),
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
