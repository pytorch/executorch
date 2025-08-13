# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.devtools.inspector.numerical_comparator import L1Comparator


class TestL1Comparator(unittest.TestCase):
    l1_comparator = L1Comparator()

    def test_identical_tensors(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[1, 2], [3, 4]])
        expected = 0.0
        result = self.l1_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)

    def test_scalar(self):
        a = 1
        b = 2
        expected = 1.0
        result = self.l1_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)

    def test_with_nans_replaced_with_zero(self):
        a = torch.tensor([3, 2, -1, float("nan")])
        b = torch.tensor([float("nan"), 0, -3, 1])
        expected = 8.0
        result = self.l1_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)

    def test_shape_mismatch_raises_exception(self):
        a = torch.tensor([0, 2, -1])
        b = torch.tensor([1, 0, -3, 4])
        with self.assertRaises(ValueError):
            self.l1_comparator.compare(a, b)

    def test_2D_tensors(self):
        a = torch.tensor([[4, 9], [6, 4]])
        b = torch.tensor([[1, 2], [3, 5]])
        expected = 14.0
        result = self.l1_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)
