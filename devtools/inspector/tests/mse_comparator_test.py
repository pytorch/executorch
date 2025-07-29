# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.devtools.inspector.numerical_comparator import MSEComparator


class TestMSEComparator(unittest.TestCase):
    mse_comparator = MSEComparator()

    def test_identical_tensors(self):
        a = torch.tensor([[10, 4], [3, 4]])
        b = torch.tensor([[10, 4], [3, 4]])
        expected = 0.0
        result = self.mse_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)

    def test_scalar(self):
        a = 10
        b = 2
        expected = 64.0
        result = self.mse_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)

    def test_with_nans_replaced_with_zero(self):
        a = torch.tensor([3, 1, -3, float("nan")])
        b = torch.tensor([float("nan"), 0, -3, 2])
        expected = (9.0 + 1.0 + 0.0 + 4.0) / 4.0
        result = self.mse_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)

    def test_shape_mismatch_raises_exception(self):
        a = torch.tensor([0, 2, -1])
        b = torch.tensor([1, 1, -3, 4])
        with self.assertRaises(ValueError):
            self.mse_comparator.compare(a, b)

    def test_2D_tensors(self):
        a = torch.tensor([[4, 9], [6, 4]])
        b = torch.tensor([[1, 2], [3, 10]])
        expected = (9.0 + 49.0 + 9.0 + 36.0) / 4.0
        result = self.mse_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)
