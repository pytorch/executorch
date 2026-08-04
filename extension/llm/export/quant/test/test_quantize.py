# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/quantize.py."""

import unittest

import torch
import torch.nn as nn

from executorch.extension.llm.export.int4 import ExportableInt4Tensor
from executorch.extension.llm.export.quant.convert import to_default
from executorch.extension.llm.export.quant.quantize import (
    dequantize_weight,
    quantize_model,
    quantize_stream,
    quantize_weight,
)
from executorch.extension.llm.export.quant.recipe import (
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
        with abs_mean ~0.01 (small-magnitude weights). Our _to_intx_tensor
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

        # quantize_model returns the in-memory export form: int4 is wrapped as
        # ExportableInt4Tensor (not raw torchao Int4Tensor).
        self.assertIsInstance(state["proj.weight"], ExportableInt4Tensor)
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


class TestQuantizeStream(unittest.TestCase):
    """quantize_stream is the lazy, per-weight dual of quantize_model."""

    def _model_and_recipe(self):
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
        return model, recipe

    def test_is_lazy(self):
        """Returns an iterator; nothing is quantized until consumed."""
        _model, recipe = self._model_and_recipe()
        gen = quantize_stream(iter([]), recipe)
        self.assertTrue(hasattr(gen, "__next__"))
        with self.assertRaises(StopIteration):
            next(gen)

    def test_matches_quantize_model_params(self):
        """Streamed params reconstruct quantize_model's parameter entries.

        quantize_stream yields the serialization form (torchao-native
        ``Int4Tensor``); quantize_model yields the in-memory form
        (``ExportableInt4Tensor``). They match once the stream output is passed
        through the same ``to_default`` convert quantize_model applies.
        """
        model, recipe = self._model_and_recipe()
        ref = quantize_model(model, recipe)
        params = ((n, p.data) for n, p in model.named_parameters())
        streamed = {n: to_default(n, v) for n, v in quantize_stream(params, recipe)}

        self.assertEqual(set(streamed), {n for n, _ in model.named_parameters()})
        for name, val in streamed.items():
            r = ref[name]
            self.assertIs(type(val), type(r))
            data_names = getattr(val, "tensor_data_names", None)
            if data_names:  # quantized subclass: compare payload bit-exactly
                for n in data_names:
                    self.assertTrue(torch.equal(getattr(val, n), getattr(r, n)))
            else:
                self.assertTrue(torch.equal(val, r))

    def test_unmatched_cast_to_dtype(self):
        recipe = QuantRecipe(rules=[QuantRule(r".*", None)])
        out = dict(
            quantize_stream(
                [("x", torch.randn(4, 4, dtype=torch.float32))],
                recipe,
                dtype=torch.float16,
            )
        )
        self.assertEqual(out["x"].dtype, torch.float16)

    def test_non_float_passthrough_uncast(self):
        """A routed non-float tensor (e.g. an int position table) is not cast."""
        recipe = QuantRecipe(rules=[QuantRule(r".*", None)])
        positions = torch.arange(8, dtype=torch.long)
        out = dict(quantize_stream([("pos", positions)], recipe, dtype=torch.bfloat16))
        self.assertEqual(out["pos"].dtype, torch.long)
        self.assertTrue(torch.equal(out["pos"], positions))


if __name__ == "__main__":
    unittest.main()
