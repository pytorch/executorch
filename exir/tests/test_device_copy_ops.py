# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

# Import the registry to register the ops
import executorch.exir.passes._device_copy_ops_registry  # noqa: F401

import torch


class DeviceCopyOpsRegistryTest(unittest.TestCase):
    """Tests that et_copy._h2d_copy and et_copy._d2h_copy ops are correctly
    registered and produce expected outputs during tracing (CPU-only)."""

    def test_h2d_copy_functional(self):
        """_h2d_copy should return a clone of the input tensor."""
        x = torch.randn(2, 3)
        result = torch.ops.et_copy._h2d_copy(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertTrue(torch.equal(result, x))
        # Should be a new tensor, not the same object
        self.assertFalse(result.data_ptr() == x.data_ptr())

    def test_d2h_copy_functional(self):
        """_d2h_copy should return a clone of the input tensor."""
        x = torch.randn(4, 5)
        result = torch.ops.et_copy._d2h_copy(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertTrue(torch.equal(result, x))
        self.assertFalse(result.data_ptr() == x.data_ptr())

    def test_h2d_copy_out_variant(self):
        """_h2d_copy.out should copy data into the provided out tensor."""
        x = torch.randn(3, 3)
        out = torch.empty(3, 3)
        result = torch.ops.et_copy._h2d_copy.out(x, out=out)
        self.assertTrue(result is out)
        self.assertTrue(torch.equal(out, x))

    def test_d2h_copy_out_variant(self):
        """_d2h_copy.out should copy data into the provided out tensor."""
        x = torch.randn(2, 4)
        out = torch.empty(2, 4)
        result = torch.ops.et_copy._d2h_copy.out(x, out=out)
        self.assertTrue(result is out)
        self.assertTrue(torch.equal(out, x))

    def test_h2d_copy_preserves_dtype(self):
        """_h2d_copy should work with various dtypes."""
        for dtype in [torch.float32, torch.float16, torch.int32, torch.int64]:
            x = torch.ones(2, 2, dtype=dtype)
            result = torch.ops.et_copy._h2d_copy(x)
            self.assertEqual(result.dtype, dtype)
            self.assertTrue(torch.equal(result, x))

    def test_h2d_copy_scalar_tensor(self):
        """_h2d_copy should handle 0-dim tensors."""
        x = torch.tensor(3.14)
        result = torch.ops.et_copy._h2d_copy(x)
        self.assertEqual(result.shape, torch.Size([]))
        self.assertTrue(torch.equal(result, x))

    def test_d2h_copy_empty_tensor(self):
        """_d2h_copy should handle empty tensors."""
        x = torch.empty(0, 3)
        result = torch.ops.et_copy._d2h_copy(x)
        self.assertEqual(result.shape, torch.Size([0, 3]))
