# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch

from executorch.devtools.inspector.numerical_comparator import SNRComparator


class TestSNRComparator(unittest.TestCase):
    snr_comparator = SNRComparator()

    def test_identical_tensors(self):
        # identical tensors --> error_power == 0 --> SNR is inf
        a = torch.tensor([[10, 4], [3, 4]])
        b = torch.tensor([[10, 4], [3, 4]])
        result = self.snr_comparator.compare(a, b)
        self.assertTrue(math.isinf(result) and result > 0)

    def test_scalar(self):
        # original_power == 1, error_power == 1 --> SNR = 10 * log10(1/1) = 0
        a = 1
        b = 2
        result = self.snr_comparator.compare(a, b)
        self.assertAlmostEqual(result, 0.0)

    def test_with_nans_replaced_with_zero(self):
        a = torch.tensor([float("nan"), 1.0])
        b = torch.tensor([0.0, 1.0])
        result = self.snr_comparator.compare(a, b)
        self.assertTrue(math.isinf(result) and result > 0)

    def test_shape_mismatch_raises_exception(self):
        a = torch.tensor([1, 2, -1])
        b = torch.tensor([1, 1, -3, 4])
        with self.assertRaises(ValueError):
            self.snr_comparator.compare(a, b)

    def test_2D_tensors(self):
        # original_power = mean([16, 81, 36, 16]) = 37.25
        # error = a - b = [3, 7, 3, -1] squared = [9, 49, 9, 1] mean = 68/4 = 17.0
        # SNR = 10 * log10(37.25/17.0)
        a = torch.tensor([[4, 9], [6, 4]])
        b = torch.tensor([[1, 2], [3, 5]])
        expected = 10 * math.log10(37.25 / 17.0)
        result = self.snr_comparator.compare(a, b)
        self.assertAlmostEqual(result, expected)
