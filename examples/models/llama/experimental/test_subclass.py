# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from .subclass import from_float, to_float


class TestGGMLTensorSubclass(unittest.TestCase):
    def test_from_float_and_to_float(self) -> None:
        weight = torch.randn([1, 32], dtype=torch.float16)

        packed = from_float(weight)

        self.assertEqual(packed.dtype, torch.uint8)
        # expected size = (1 * sizeof(uint16_t)) + (32 * sizeof(uint8_t)/2) = 18
        self.assertEqual(packed.numel(), 18)

        unpacked = to_float(packed).reshape(weight.shape)

        tolerance = (torch.max(weight) - torch.min(weight)) / 16

        self.assertTrue(torch.allclose(weight, unpacked, atol=tolerance.item()))
