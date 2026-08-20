# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/quantize.py."""

import unittest

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.quant.quantize import (
    dequantize_weight,
    quantize_model,
    quantize_weight,
)
from executorch.examples.models.gemma4_31b.quant.recipe import (
    QuantConfig,
    QuantRecipe,
    QuantRule,
)
from parameterized import parameterized
from torchao.quantization import IntxUnpackedToInt8Tensor
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor


class TestQuantizeWeight(unittest.TestCase):
    @parameterized.expand(
        [
            ("4bit_asym", 4, 32, False),
            ("4bit_sym", 4, 32, True),
            ("4bit_gs64", 4, 64, False),
            ("8bit_sym", 8, 32, True),
        ]
    )
    def test_output_type(self, _name, bits, gs, sym):
        config = QuantConfig(bits=bits, group_size=gs, symmetric=sym, method="min_max")
        result = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        if bits == 4:
            self.assertIsInstance(result, Int4Tensor)
            self.assertEqual(result.shape, torch.Size([64, 128]))
        else:
            self.assertIsInstance(result, IntxUnpackedToInt8Tensor)
            self.assertEqual(result.shape, torch.Size([64, 128]))

    def test_quantize_dequantize_roundtrip(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 128, dtype=torch.bfloat16)
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(weight, config)
        dequant = dequantize_weight(q, dtype=torch.bfloat16)
        rel_error = (
            dequant.float() - weight.float()
        ).abs().mean() / weight.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_dequantize_output_dtype(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)
        self.assertEqual(dequantize_weight(q, torch.float32).dtype, torch.float32)
        self.assertEqual(dequantize_weight(q, torch.bfloat16).dtype, torch.bfloat16)

    def test_dequantize_symmetric_4bit(self):
        torch.manual_seed(1)
        weight = torch.randn(32, 64, dtype=torch.bfloat16)
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        dequant = dequantize_weight(q)
        self.assertEqual(dequant.shape, (32, 64))
        rel_error = (
            dequant.float() - weight.float()
        ).abs().mean() / weight.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_dequantize_int8(self):
        torch.manual_seed(2)
        weight = torch.randn(32, 64, dtype=torch.bfloat16)
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        dequant = dequantize_weight(q, dtype=torch.bfloat16)
        rel_error = (
            dequant.float() - weight.float()
        ).abs().mean() / weight.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_int8_small_weights_bf16_precision(self):
        """INT8 quantization of small bf16 weights must use full int8 range.

        Regression: IntxUnpackedToInt8Tensor.from_hp quantizes in bf16,
        which collapses per-group scales to a single value for weights
        with abs_mean ~0.01 (e.g., Gemma 4 v_proj). Our _to_intx_tensor
        casts to float32 first to avoid this.
        """
        torch.manual_seed(42)
        weight = torch.randn(64, 128, dtype=torch.bfloat16) * 0.01
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        dequant = dequantize_weight(q, dtype=torch.bfloat16)
        rel_error = (
            dequant.float() - weight.float()
        ).abs().mean() / weight.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_dequantize_int8_asymmetric(self):
        torch.manual_seed(3)
        weight = torch.randn(32, 64, dtype=torch.bfloat16)
        config = QuantConfig(bits=8, group_size=32, symmetric=False, method="min_max")
        q = quantize_weight(weight, config)
        dequant = dequantize_weight(q, dtype=torch.bfloat16)
        rel_error = (
            dequant.float() - weight.float()
        ).abs().mean() / weight.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)

    def test_int8_per_axis(self):
        """Per-axis (group_size == K) used for embeddings."""
        weight = torch.randn(256, 64, dtype=torch.bfloat16)
        config = QuantConfig(bits=8, group_size=64, symmetric=True, method="min_max")
        q = quantize_weight(weight, config)
        self.assertIsInstance(q, IntxUnpackedToInt8Tensor)
        dequant = dequantize_weight(q, dtype=torch.bfloat16)
        rel_error = (
            dequant.float() - weight.float()
        ).abs().mean() / weight.float().abs().mean()
        self.assertLess(rel_error.item(), 0.01)

    @parameterized.expand(
        [
            ("unknown_method", QuantConfig(4, 32, False, "bogus"), "bogus"),
            ("unsupported_bits", QuantConfig(3, 32, False, "min_max"), None),
            ("hqq_8bit_asym", QuantConfig(8, 32, False, "hqq"), "symmetric"),
        ]
    )
    def test_invalid_config_raises(self, _name, config, expected_substr):
        with self.assertRaises(ValueError) as ctx:
            quantize_weight(torch.randn(32, 64), config)
        if expected_substr:
            self.assertIn(expected_substr, str(ctx.exception))


class TestQuantizeWeightHQQ(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for HQQ")

    def test_quantize_dequantize_roundtrip(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="hqq")
        q = quantize_weight(weight, config)
        dequant = dequantize_weight(q, dtype=torch.bfloat16).cpu()
        rel_error = (
            dequant.float() - weight.cpu().float()
        ).abs().mean() / weight.cpu().float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_symmetric_scale_only(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="hqq")
        q = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        self.assertIsInstance(q, Int4Tensor)

    def test_int8_hqq_roundtrip(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 128, dtype=torch.bfloat16)
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="hqq")
        q = quantize_weight(weight, config)
        self.assertIsInstance(q, IntxUnpackedToInt8Tensor)
        dequant = dequantize_weight(q, dtype=torch.bfloat16)
        rel_error = (
            dequant.float() - weight.float()
        ).abs().mean() / weight.float().abs().mean()
        self.assertLess(rel_error.item(), 0.02)


class TestQuantizeModel(unittest.TestCase):
    def test_applies_recipe(self):
        model = nn.ModuleDict(
            {
                "embed": nn.Embedding(32, 16),
                "proj": nn.Linear(16, 32, bias=False),
                "norm": nn.LayerNorm(32),
            }
        )
        model.to(dtype=torch.bfloat16)
        for p in model.parameters():
            p.data.normal_(0, 0.02)

        recipe = QuantRecipe(
            rules=[
                QuantRule(r"embed\.weight", None),
                QuantRule(r"norm\.weight", None),
                QuantRule(r".*\.weight", QuantConfig(4, 16, False, "min_max")),
            ]
        )
        state = quantize_model(model, recipe)

        self.assertIsInstance(state["proj.weight"], Int4Tensor)
        self.assertIs(type(state["embed.weight"]), torch.Tensor)
        self.assertIs(type(state["norm.weight"]), torch.Tensor)

    def test_persistent_buffers_included(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.randn(16, 32, dtype=torch.bfloat16))
        model.register_buffer("scalar", torch.ones(1))
        model.register_buffer("temp", torch.zeros(4), persistent=False)

        recipe = QuantRecipe(rules=[QuantRule(r".*", None)])
        state = quantize_model(model, recipe)

        self.assertIn("scalar", state)
        self.assertNotIn("temp", state)

    def test_unquantized_cast_to_dtype(self):
        model = nn.ModuleDict({"proj": nn.Linear(16, 8, bias=False)})
        model.proj.weight.data = torch.randn(8, 16, dtype=torch.float32)

        recipe = QuantRecipe(rules=[QuantRule(r".*", None)])
        state = quantize_model(model, recipe, dtype=torch.float16)

        self.assertEqual(state["proj.weight"].dtype, torch.float16)

    def test_empty_model(self):
        state = quantize_model(nn.Module(), QuantRecipe(rules=[]))
        self.assertEqual(len(state), 0)


if __name__ == "__main__":
    unittest.main()
