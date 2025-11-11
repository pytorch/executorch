#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.kernels.quantized  # noqa[F401] 'executorch.kernels.quantized' imported but unused

import torch
import torch.ao.quantization.fx._decomposed  # noqa[F401] 'torch.ao.quantization.fx._decomposed' imported but unused
from executorch.exir.dialects._ops import ops
from executorch.exir.passes._quant_patterns_and_replacements import (  # noqa
    quantized_decomposed_lib,  # noqa
)


class TestOutVariants(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_add_to_out_variant(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.add.out)
        fn = ops.edge.quantized_decomposed.add.default
        out_variant = fn.to_out_variant()
        self.assertEqual(out_variant.name(), "quantized_decomposed::add.out")

    def test_choose_qparams_tensor_to_out_variant(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.choose_qparams.Tensor_out)
        choose_qparams = ops.edge.quantized_decomposed.choose_qparams.tensor
        out_variant = choose_qparams.to_out_variant()
        self.assertEqual(
            out_variant.name(), "quantized_decomposed::choose_qparams.Tensor_out"
        )

    def test_dequantize_per_tensor_to_out_variant(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.dequantize_per_tensor.out)
        fn = ops.edge.quantized_decomposed.dequantize_per_tensor.default
        out_variant = fn.to_out_variant()
        self.assertEqual(
            out_variant.name(), "quantized_decomposed::dequantize_per_tensor.out"
        )

    def test_dequantize_per_tensor_tensor_to_out_variant(self) -> None:
        self.assertIsNotNone(
            ops.edge.quantized_decomposed.dequantize_per_tensor.Tensor_out
        )
        fn = ops.edge.quantized_decomposed.dequantize_per_tensor.tensor
        out_variant = fn.to_out_variant()
        self.assertEqual(
            out_variant.name(), "quantized_decomposed::dequantize_per_tensor.Tensor_out"
        )

    def test_dequantize_per_channel_to_out_variant(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.dequantize_per_channel.out)
        fn = ops.edge.quantized_decomposed.dequantize_per_channel.default
        out_variant = fn.to_out_variant()
        self.assertEqual(
            out_variant.name(), "quantized_decomposed::dequantize_per_channel.out"
        )

    def test_mixed_linear_to_out_variant(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.mixed_linear.out)
        fn = ops.edge.quantized_decomposed.mixed_linear.default
        out_variant = fn.to_out_variant()
        self.assertEqual(out_variant.name(), "quantized_decomposed::mixed_linear.out")

    def test_mixed_mm_to_out_variant(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.mixed_mm.out)
        fn = ops.edge.quantized_decomposed.mixed_mm.default
        out_variant = fn.to_out_variant()
        self.assertEqual(out_variant.name(), "quantized_decomposed::mixed_mm.out")

    def test_quantize_per_tensor_to_out_variant(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.quantize_per_tensor.out)
        fn = ops.edge.quantized_decomposed.quantize_per_tensor.default
        out_variant = fn.to_out_variant()
        self.assertEqual(
            out_variant.name(), "quantized_decomposed::quantize_per_tensor.out"
        )

    def test_quantize_per_tensor_tensor_to_out_variant(self) -> None:
        self.assertIsNotNone(
            ops.edge.quantized_decomposed.quantize_per_tensor.Tensor_out
        )
        fn = ops.edge.quantized_decomposed.quantize_per_tensor.tensor
        out_variant = fn.to_out_variant()
        self.assertEqual(
            out_variant.name(), "quantized_decomposed::quantize_per_tensor.Tensor_out"
        )

    def test_quantize_per_channel_to_out_variant(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.quantize_per_channel.out)
        fn = ops.edge.quantized_decomposed.quantize_per_channel.default
        out_variant = fn.to_out_variant()
        self.assertEqual(
            out_variant.name(), "quantized_decomposed::quantize_per_channel.out"
        )
