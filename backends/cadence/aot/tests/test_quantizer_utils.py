# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.backends.cadence.aot.quantizer.utils import (
    check_out_zero_point_is_min_range,
)


class TestCheckOutZeroPointIsMinRange(unittest.TestCase):
    def test_signed_types(self) -> None:
        self.assertTrue(check_out_zero_point_is_min_range(-128, torch.int8))
        self.assertFalse(check_out_zero_point_is_min_range(0, torch.int8))
        self.assertTrue(check_out_zero_point_is_min_range(-32768, torch.int16))
        self.assertFalse(check_out_zero_point_is_min_range(0, torch.int16))

    def test_unsigned_types(self) -> None:
        self.assertTrue(check_out_zero_point_is_min_range(0, torch.uint8))
        self.assertFalse(check_out_zero_point_is_min_range(5, torch.uint8))
        self.assertTrue(check_out_zero_point_is_min_range(0, torch.uint16))

    def test_non_quant_dtype_is_false(self) -> None:
        # Dtypes that are not one of the handled quant types must return False,
        # regardless of the zero point value (regression for an `or` precedence
        # bug that made the unsigned branch always match).
        self.assertFalse(check_out_zero_point_is_min_range(0, torch.float32))
        self.assertFalse(check_out_zero_point_is_min_range(0, torch.int32))


if __name__ == "__main__":
    unittest.main()
