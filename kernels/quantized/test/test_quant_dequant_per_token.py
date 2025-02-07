# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401


class QuantizePerTokenTest(unittest.TestCase):

    def test_quantize_per_token(self):
        input_tensor = torch.tensor(
            [[-0.5, 0.3, 1.2], [0.1, -0.8, 2.1], [-5, 1, 2]], dtype=torch.float32
        )
        scale = torch.tensor([0.5, 0.8, 1.0], dtype=torch.float64)
        scale = scale.unsqueeze(-1)
        zero_point = torch.tensor([-1, -2, 0])
        zero_point = zero_point.unsqueeze(-1)
        quantized_tensor = torch.ops.quantized_decomposed.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )
        expected_quantized_tensor = torch.ops.et_quant_test.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )

        self.assertTrue(torch.equal(quantized_tensor, expected_quantized_tensor))

    def test_quantize_per_token_large_tensor(self):
        input_tensor = torch.rand((8, 32))
        scale = torch.rand((8, 1), dtype=torch.float64)
        zero_point = torch.randint(0, 10, (8, 1))
        quantized_tensor = torch.ops.quantized_decomposed.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )
        expected_quantized_tensor = torch.ops.et_quant_test.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )

        self.assertTrue(torch.equal(quantized_tensor, expected_quantized_tensor))

    def test_quantize_per_token_high_rank(self):
        input_tensor = torch.rand((1, 3, 8, 32))
        scale = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        zero_point = torch.randint(0, 10, (1, 3, 8, 1))
        quantized_tensor = torch.ops.quantized_decomposed.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )
        expected_quantized_tensor = torch.ops.et_quant_test.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )

        self.assertTrue(torch.equal(quantized_tensor, expected_quantized_tensor))

    def test_quantize_per_token_dynamic(self):
        input_tensor = torch.rand((1, 1, 8, 1))
        scale = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        zero_point = torch.randint(0, 10, (1, 1, 8, 1))
        quantized_tensor = torch.ops.quantized_decomposed.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )
        expected_quantized_tensor = torch.ops.et_quant_test.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )

        self.assertTrue(torch.equal(quantized_tensor, expected_quantized_tensor))

        input_tensor = torch.rand((1, 3, 8, 1))
        scale = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        zero_point = torch.randint(0, 10, (1, 3, 8, 1))
        quantized_tensor = torch.ops.quantized_decomposed.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )
        expected_quantized_tensor = torch.ops.et_quant_test.quantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8
        )

        self.assertTrue(torch.equal(quantized_tensor, expected_quantized_tensor))

    def test_dequantize_per_token(self):
        input_tensor = torch.randint(-50, 120, (3, 3), dtype=torch.int8)
        scale = torch.tensor([0.5, 0.8, 1.0], dtype=torch.float64)
        scale = scale.unsqueeze(-1)
        zero_point = torch.tensor([-1, -2, 0])
        zero_point = zero_point.unsqueeze(-1)
        dequantized_tensor = torch.ops.quantized_decomposed.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )
        expected_dequantized_tensor = torch.ops.et_quant_test.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )

        self.assertTrue(torch.allclose(dequantized_tensor, expected_dequantized_tensor))

    def test_dequantize_per_token_large_tensor(self):
        input_tensor = torch.randint(-50, 120, (8, 32), dtype=torch.int8)
        scale = torch.rand((8, 1), dtype=torch.float64)
        zero_point = torch.randint(0, 10, (8, 1))
        dequantized_tensor = torch.ops.quantized_decomposed.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )
        expected_dequantized_tensor = torch.ops.et_quant_test.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )

        self.assertTrue(torch.allclose(dequantized_tensor, expected_dequantized_tensor))

    def test_dequantize_per_token_high_rank(self):
        input_tensor = torch.randint(-50, 120, (1, 3, 8, 32), dtype=torch.int8)
        scale = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        zero_point = torch.randint(0, 10, (1, 3, 8, 1))
        dequantized_tensor = torch.ops.quantized_decomposed.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )
        expected_dequantized_tensor = torch.ops.et_quant_test.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )

        self.assertTrue(torch.allclose(dequantized_tensor, expected_dequantized_tensor))

    def test_dequantize_per_token_dynamic(self):
        input_tensor = torch.randint(-50, 120, (1, 1, 8, 32), dtype=torch.int8)
        scale = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        zero_point = torch.randint(0, 10, (1, 1, 8, 1))
        dequantized_tensor = torch.ops.quantized_decomposed.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )
        expected_dequantized_tensor = torch.ops.et_quant_test.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )

        self.assertTrue(torch.allclose(dequantized_tensor, expected_dequantized_tensor))

        input_tensor = torch.randint(-50, 120, (1, 3, 8, 32), dtype=torch.int8)
        scale = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        zero_point = torch.randint(0, 10, (1, 3, 8, 1))
        dequantized_tensor = torch.ops.quantized_decomposed.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )
        expected_dequantized_tensor = torch.ops.et_quant_test.dequantize_per_token(
            input_tensor, scale, zero_point, -128, 127, torch.int8, torch.float32
        )

        self.assertTrue(torch.allclose(dequantized_tensor, expected_dequantized_tensor))
