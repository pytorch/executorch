# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[6]
# pyre-ignore-all-errors[16]
import unittest

from typing import List, Optional

import executorch.exir.schema as schema

import torch
from executorch.exir.tensor import (
    contiguous_stride_from_shape,
    dim_order_from_stride,
    make_allocation_info,
    make_tensor_value,
    num_bytes_from_shape_and_dtype,
    scalar_type_enum,
    stride_from_dim_order,
    TensorSpec,
)


class TestTensor(unittest.TestCase):
    def compare_tensors(
        self,
        torch_tensor: torch.Tensor,
        flatbuffer_tensor: schema.Tensor,
        dim_order: Optional[List[int]] = None,
    ) -> None:
        """Checks if the given normal torch tensor is equivalent to the
        flatbuffer tensor.
        """
        self.assertEqual(
            flatbuffer_tensor.scalar_type, scalar_type_enum(torch_tensor.dtype)
        )
        # The runtime currently only supports tensors with offset 0.
        self.assertEqual(flatbuffer_tensor.storage_offset, 0)
        self.assertEqual(flatbuffer_tensor.sizes, list(torch_tensor.size()))
        self.assertEqual(flatbuffer_tensor.requires_grad, torch_tensor.requires_grad)
        if dim_order is not None:
            self.assertEqual(flatbuffer_tensor.dim_order, dim_order)

    def test_normal_tensor_conversion(self) -> None:
        """Testing a normal tensor"""

        normal_tensor = torch.randn(2, 2, 3)
        flatbuffer_tensor = make_tensor_value(
            1, 0, TensorSpec.from_tensor(normal_tensor)
        )
        self.compare_tensors(normal_tensor, flatbuffer_tensor)

        # Test zero size tensor
        normal_tensor = torch.randn(2, 2, 0)
        flatbuffer_tensor = make_tensor_value(
            1, 0, TensorSpec.from_tensor(normal_tensor)
        )
        self.compare_tensors(normal_tensor, flatbuffer_tensor)

        # Test zero size tensor
        normal_tensor = torch.randn(2, 0, 3)
        flatbuffer_tensor = make_tensor_value(
            1, 0, TensorSpec.from_tensor(normal_tensor)
        )
        self.compare_tensors(normal_tensor, flatbuffer_tensor)

        # Test zero size tensor
        normal_tensor = torch.randn(0, 2, 3)
        flatbuffer_tensor = make_tensor_value(
            1, 0, TensorSpec.from_tensor(normal_tensor)
        )
        self.compare_tensors(normal_tensor, flatbuffer_tensor)

        # Compare dim order
        normal_tensor = torch.rand((2, 2, 3, 4))
        flatbuffer_tensor = make_tensor_value(
            1, 0, TensorSpec.from_tensor(normal_tensor)
        )
        self.compare_tensors(normal_tensor, flatbuffer_tensor, dim_order=[0, 1, 2, 3])
        # cannot compare torch.memory_format = torch.channels_last because make_tensor_value
        # infers strides from sizes assuming tensor dimensions are laid out in memory
        # in the same order as indicated by dimension order of sizes array.
        # e.g. for sizes = (2, 3, 4, 5), it assumes dimension order is (0, 1, 2, 3) and
        # thus strides = (3*4*5, 4*5, 5, 1)
        # whereas strides for torch.memory_format = torch.channels_last is
        # (3*4*5, 1, 5*3, 3))

    def test_allocation_info_succeeds(self) -> None:
        test_cases = (
            (
                {"mem_id": 0, "mem_offset": 0},
                schema.AllocationDetails(
                    memory_id=0, memory_offset_low=0, memory_offset_high=0
                ),
            ),
            (
                # Easily fits in 32 bits
                {"mem_id": 1, "mem_offset": 55555},
                schema.AllocationDetails(
                    memory_id=1, memory_offset_low=55555, memory_offset_high=0
                ),
            ),
            (
                # Just fits in 32 bits
                {"mem_id": 1, "mem_offset": (1 << 32) - 1},
                schema.AllocationDetails(
                    memory_id=1, memory_offset_low=0xFFFFFFFF, memory_offset_high=0
                ),
            ),
            (
                # Smallest 32-bit overflow.
                {"mem_id": 1, "mem_offset": 1 << 32},
                schema.AllocationDetails(
                    memory_id=1, memory_offset_low=0, memory_offset_high=1
                ),
            ),
            (
                # Easily fits in 64 bits.
                {"mem_id": 1, "mem_offset": (1 << 64) - 55555555},
                schema.AllocationDetails(
                    memory_id=1,
                    memory_offset_low=4239411741,
                    memory_offset_high=4294967295,
                ),
            ),
            (
                # Just fits in 64 bits
                {"mem_id": 1, "mem_offset": (1 << 64) - 1},
                schema.AllocationDetails(
                    memory_id=1,
                    memory_offset_low=0xFFFFFFFF,
                    memory_offset_high=0xFFFFFFFF,
                ),
            ),
        )
        for test_case in test_cases:
            allocation_info = make_allocation_info(**(test_case[0]))
            self.assertEqual(allocation_info, test_case[1])

    def test_allocation_info_fails(self) -> None:
        test_cases = (
            (
                # Significant negative underflow.
                {"mem_id": 0, "mem_offset": -55555},
                # Error message should complain about the negative value.
                "negative",
            ),
            (
                # Smallest negative underflow.
                {"mem_id": 0, "mem_offset": -1},
                # Error message should complain about the negative value.
                "negative",
            ),
            (
                # Smallest 64-bit overflow.
                {"mem_id": 1, "mem_offset": 1 << 64},
                # Error message should complain that the value is too large.
                "64 bits",
            ),
            (
                # Significant 64-bit overflow.
                {"mem_id": 1, "mem_offset": (1 << 64) + 55555},
                # Error message should complain that the value is too large.
                "64 bits",
            ),
        )
        for test_case in test_cases:
            kwargs = test_case[0]
            with self.assertRaisesRegex(Exception, test_case[1], msg=f"{kwargs}"):
                make_allocation_info(**kwargs)

    def test_contiguous_stride_from_shape(self) -> None:
        shape = (2, 3, 4)
        stride = contiguous_stride_from_shape(torch.Size(shape))
        self.assertEqual((12, 4, 1), stride)

    def test_dim_order_from_stride(self) -> None:
        # shape = (4)
        strides = (1,)
        dim_order = dim_order_from_stride(strides)
        print(dim_order)
        self.assertEqual((0,), dim_order)

        # Test contiguous, a.k.a NCHW format
        # shape = (2, 3, 4)
        strides = (3 * 4, 4, 1)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 1, 2), dim_order)

        # shape = (2, 3, 4, 5)
        strides = (3 * 4 * 5, 4 * 5, 5, 1)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 1, 2, 3), dim_order)

        # shape = (2, 3, 4, 5, 6)
        strides = (3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 1, 2, 3, 4), dim_order)

        # Test channels last format
        # shape = (2, 3, 4)
        strides = (3 * 4, 1, 3)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 2, 1), dim_order)

        # shape = (2, 3, 4, 5)
        strides = (3 * 4 * 5, 1, 5 * 3, 3)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 2, 3, 1), dim_order)

        # shape = (2, 3, 4, 5, 6)
        strides = (3 * 4 * 5 * 6, 1, 5 * 6 * 3, 6 * 3, 3)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 2, 3, 4, 1), dim_order)

        # test ambiguous strides
        # shape = (1, 3, 3, 1)
        strides = (9, 3, 1, 1)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 1, 2, 3), dim_order)

        # test ambiguous strides
        # shape = (1, 3, 1, 1)
        strides = (3, 1, 3, 3)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 2, 3, 1), dim_order)

        # test ambiguous strides
        # shape = (1, 3, 1, 1)
        strides = (3, 1, 1, 1)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 1, 2, 3), dim_order)

        # test ambiguous strides
        # shape = (1, 1, 1, 1)
        strides = (1, 1, 1, 1)
        dim_order = dim_order_from_stride(strides)
        self.assertEqual((0, 1, 2, 3), dim_order)

        # test 0 in strides
        # dim[2] is broadcasting dim
        # shape = (5, 1, 15, 10)
        strides = (10, 10, 0, 1)
        with self.assertRaises(ValueError):
            dim_order = dim_order_from_stride(strides)

    def test_strides_from_dim_order(self) -> None:
        sizes = []
        dim_order = []
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual([], strides)

        sizes = [
            4,
        ]
        dim_order = [
            0,
        ]
        expected_strides = [
            1,
        ]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        # Test contiguous, a.k.a NCHW format
        sizes = [2, 3, 4]
        dim_order = [0, 1, 2]
        expected_strides = [3 * 4, 4, 1]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        sizes = [2, 3, 4, 5]
        dim_order = [0, 1, 2, 3]
        expected_strides = [3 * 4 * 5, 4 * 5, 5, 1]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        sizes = [2, 3, 4, 5, 6]
        dim_order = [0, 1, 2, 3, 4]
        expected_strides = [3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        # Test channels last format
        sizes = [2, 3, 4]
        dim_order = [0, 2, 1]
        expected_strides = [3 * 4, 1, 3]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        sizes = [2, 3, 4, 5]
        dim_order = [0, 2, 3, 1]
        expected_strides = [3 * 4 * 5, 1, 5 * 3, 3]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        sizes = [2, 3, 4, 5, 6]
        dim_order = [0, 2, 3, 4, 1]
        expected_strides = [3 * 4 * 5 * 6, 1, 5 * 6 * 3, 6 * 3, 3]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        # test ambiguous strides
        sizes = [1, 3, 3, 1]
        dim_order = [0, 1, 2, 3]
        expected_strides = [9, 3, 1, 1]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        # test ambiguous strides
        sizes = [1, 3, 1, 1]
        dim_order = [0, 2, 3, 1]
        expected_strides = [3, 1, 3, 3]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        # test ambiguous strides
        sizes = [1, 3, 1, 1]
        dim_order = [0, 1, 2, 3]
        expected_strides = [3, 1, 1, 1]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

        # test ambiguous strides
        sizes = [1, 1, 1, 1]
        dim_order = [0, 1, 2, 3]
        expected_strides = [1, 1, 1, 1]
        strides = stride_from_dim_order(sizes, dim_order)
        self.assertEqual(expected_strides, strides)

    def test_num_bytes_from_shape_and_dtype(self) -> None:
        shape = (2, 3, 4)
        self.assertEqual(24, num_bytes_from_shape_and_dtype(shape, torch.int8))
        self.assertEqual(48, num_bytes_from_shape_and_dtype(shape, torch.half))
        self.assertEqual(96, num_bytes_from_shape_and_dtype(shape, torch.float))
        self.assertEqual(192, num_bytes_from_shape_and_dtype(shape, torch.float64))
