# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch

from executorch.backends.cadence.aot.ref_implementations import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from executorch.backends.cadence.aot.typing_stubs import expand


class TestRefImplementations(unittest.TestCase):
    @expand(
        [
            ("basic_int8", 0.42, -1.0, 2.0, -7, 7, torch.int8, 0),
            ("basic_int16", 0.42, -1.0, 5.0, -6, 7, torch.int16, -3),
        ]
    )
    def test_quantize_per_tensor(
        self,
        name: str,
        input_value: float,
        f_min: float,
        f_max: float,
        q_min: int,
        q_max: int,
        target_dtype: torch.dtype,
        expected_value: int,
    ) -> None:
        input_tensor = torch.tensor([input_value])
        scale = (f_max - f_min) / (q_max - q_min)
        zero_point = round(-f_min / scale) + q_min
        expected_output = torch.tensor([expected_value], dtype=target_dtype)

        output = quantize_per_tensor(
            input_tensor, scale, zero_point, q_min, q_max, target_dtype
        )

        self.assertEqual(
            output.dtype, expected_output.dtype, f"Dtype mismatch in {name}"
        )
        self.assertTrue(
            torch.equal(output, expected_output),
            f"Values don't match in {name}: got {output}, expected {expected_output}",
        )

    @expand(
        [
            # Signed quantization ranges
            ("signed_range0_int8", 0, -1.0, 2.0, -7, 7, torch.int8, 0.428),
            ("signed_range0_int16", 0, -1.0, 2.0, -7, 7, torch.int16, 0.428),
            ("signed_range0_int32", 0, -1.0, 2.0, -7, 7, torch.int32, 0.428),
            ("signed_range1_int8", -3, -1.0, 5.0, -6, 7, torch.int8, 0.461),
            ("signed_range1_int16", -3, -1.0, 5.0, -6, 7, torch.int16, 0.461),
            ("signed_range1_int32", -3, -1.0, 5.0, -6, 7, torch.int32, 0.461),
            # Unsigned quantization ranges
            ("unsigned_range0_uint8", 3, -1.0, 2.0, 0, 7, torch.uint8, 0.428),
            ("unsigned_range0_uint16", 3, -1.0, 2.0, 0, 7, torch.uint16, 0.428),
            ("unsigned_range1_uint8", 4, -1.0, 5.0, 3, 7, torch.uint8, 0.0),
            ("unsigned_range1_uint16", 4, -1.0, 5.0, 3, 7, torch.uint16, 0.0),
        ]
    )
    def test_dequantize_per_tensor(
        self,
        name: str,
        input_value: int,
        f_min: float,
        f_max: float,
        q_min: int,
        q_max: int,
        input_dtype: torch.dtype,
        expected_value: int,
    ) -> None:
        input_tensor = torch.tensor([input_value], dtype=input_dtype)
        scale = (f_max - f_min) / (q_max - q_min)
        zero_point = round(-f_min / scale) + q_min
        expected_output = torch.tensor([expected_value], dtype=torch.float32)

        output = dequantize_per_tensor(
            input_tensor, scale, zero_point, q_min, q_max, torch.float32
        )

        self.assertEqual(
            output.dtype, expected_output.dtype, f"Dtype mismatch in {name}"
        )
        self.assertTrue(
            torch.allclose(output, expected_output, rtol=0.001, atol=0.001),
            f"Values don't match in {name}: got {output}, expected {expected_output}",
        )
