# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/quantize.py.

Tests the public API: ``quantize_weight`` and ``quantize_model``. Organized
by resource requirement (CPU vs CUDA), not by internal codepath.
"""

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


# ---------------------------------------------------------------------------
# quantize_weight — CPU (uses min_max; tests the output contract)


class TestQuantizeWeight(unittest.TestCase):
    @parameterized.expand(
        [
            ("4bit_asym", 4, 32, False, (64, 128), 0, 15),
            ("4bit_sym", 4, 32, True, (64, 128), 0, 15),
            ("4bit_gs64", 4, 64, False, (32, 128), 0, 15),
            ("8bit_sym", 8, 32, True, (32, 64), -128, 127),
            ("3d_expert", 4, 32, False, (8, 64, 128), 0, 15),
        ]
    )
    def test_output_structure(self, _name, bits, gs, sym, shape, qmin, qmax):
        config = QuantConfig(bits=bits, group_size=gs, symmetric=sym, method="min_max")
        cw = quantize_weight(torch.randn(*shape, dtype=torch.bfloat16), config)

        self.assertEqual(cw.qdata.shape, shape)
        self.assertEqual(cw.qdata.dtype, torch.int8)
        self.assertEqual(cw.scale.shape, (*shape[:-1], shape[-1] // gs))
        self.assertGreaterEqual(cw.qdata.min().item(), qmin)
        self.assertLessEqual(cw.qdata.max().item(), qmax)

        if sym:
            self.assertIsNone(cw.zero)
        else:
            self.assertIsNotNone(cw.zero)
            self.assertEqual(cw.zero.shape, cw.scale.shape)

        self.assertEqual(cw.config, config)

    def test_fp32_input(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(32, 64, dtype=torch.float32), config)
        self.assertEqual(cw.qdata.shape, (32, 64))

    def test_quantize_dequantize_roundtrip(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 128, dtype=torch.bfloat16)
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(weight, config)
        dequant = dequantize_weight(cw, dtype=torch.bfloat16)
        rel_error = (
            dequant.float() - weight.float()
        ).abs().mean() / weight.float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_dequantize_output_dtype(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
        cw = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)
        self.assertEqual(dequantize_weight(cw, torch.float32).dtype, torch.float32)
        self.assertEqual(dequantize_weight(cw, torch.bfloat16).dtype, torch.bfloat16)
        self.assertEqual(dequantize_weight(cw, torch.float16).dtype, torch.float16)

    def test_dequantize_symmetric(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        cw = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)
        self.assertIsNone(cw.zero)
        dequant = dequantize_weight(cw)
        self.assertEqual(dequant.shape, (32, 64))

    @parameterized.expand(
        [
            ("unknown_method", QuantConfig(4, 32, False, "bogus"), "bogus"),
            ("unsupported_bits", QuantConfig(3, 32, False, "min_max"), None),
        ]
    )
    def test_invalid_config_raises(self, _name, config, expected_substr):
        with self.assertRaises(ValueError) as ctx:
            quantize_weight(torch.randn(32, 64), config)
        if expected_substr:
            self.assertIn(expected_substr, str(ctx.exception))


# ---------------------------------------------------------------------------
# quantize_weight — CUDA (HQQ-specific behavior only)


class TestQuantizeWeightHQQ(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for HQQ")

    def test_quantize_dequantize_roundtrip(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="hqq")
        cw = quantize_weight(weight, config)
        dequant = dequantize_weight(cw, dtype=torch.bfloat16).cpu()
        rel_error = (
            dequant.float() - weight.cpu().float()
        ).abs().mean() / weight.cpu().float().abs().mean()
        self.assertLess(rel_error.item(), 0.15)

    def test_symmetric_scale_only(self):
        """symmetric=True dispatches to scale-only HQQ (no zero)."""
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="hqq")
        cw = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        self.assertIsNone(cw.zero)
        self.assertGreaterEqual(cw.qdata.min().item(), 0)
        self.assertLessEqual(cw.qdata.max().item(), 15)

    def test_cpu_input_accepted(self):
        config = QuantConfig(bits=4, group_size=32, symmetric=False, method="hqq")
        cw = quantize_weight(torch.randn(32, 64, dtype=torch.bfloat16), config)
        self.assertEqual(cw.qdata.shape, (32, 64))


# ---------------------------------------------------------------------------
# quantize_model


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

        quantized, unquantized = quantize_model(model, recipe)

        self.assertIn("proj.weight", quantized)
        self.assertEqual(quantized["proj.weight"].qdata.shape, (32, 16))
        self.assertIn("embed.weight", unquantized)
        self.assertIn("norm.weight", unquantized)
        self.assertNotIn("embed.weight", quantized)
        self.assertNotIn("norm.weight", quantized)

    def test_persistent_buffers_included(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.randn(16, 32, dtype=torch.bfloat16))
        model.register_buffer("scalar", torch.ones(1))
        model.register_buffer("temp", torch.zeros(4), persistent=False)

        recipe = QuantRecipe(rules=[QuantRule(r".*", None)])
        _, unquantized = quantize_model(model, recipe)

        self.assertIn("scalar", unquantized)
        self.assertNotIn("temp", unquantized)

    def test_unquantized_cast_to_dtype(self):
        model = nn.ModuleDict({"proj": nn.Linear(16, 8, bias=False)})
        model.proj.weight.data = torch.randn(8, 16, dtype=torch.float32)

        recipe = QuantRecipe(rules=[QuantRule(r".*", None)])
        _, unquantized = quantize_model(model, recipe, dtype=torch.float16)

        self.assertEqual(unquantized["proj.weight"].dtype, torch.float16)

    def test_empty_model(self):
        quantized, unquantized = quantize_model(nn.Module(), QuantRecipe(rules=[]))
        self.assertEqual(len(quantized), 0)
        self.assertEqual(len(unquantized), 0)

    def test_all_quantized(self):
        model = nn.ModuleDict({"a": nn.Linear(32, 16, bias=False)})
        model.to(dtype=torch.bfloat16)
        for p in model.parameters():
            p.data.normal_(0, 0.02)

        config = QuantConfig(bits=4, group_size=16, symmetric=False, method="min_max")
        quantized, unquantized = quantize_model(
            model, QuantRecipe(rules=[QuantRule(r".*", config)])
        )
        self.assertEqual(len(quantized), 1)
        self.assertIn("a.weight", quantized)
        self.assertEqual(len(unquantized), 0)


if __name__ == "__main__":
    unittest.main()
