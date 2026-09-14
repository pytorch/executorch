# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the weight-conversion helpers (``convert.py``) and the
model-assignment helpers (``assign_one`` / ``assign_state_dict`` in ``load.py``).
No hardware required.
"""

import unittest

import torch
import torch.nn as nn

from executorch.extension.llm.export.load import assign_one, assign_state_dict
from executorch.extension.llm.export.quant.convert import (
    fuse_along_output,
    to_default,
    to_exportable,
)
from executorch.extension.llm.export.quant.quantize import (
    dequantize_weight,
    quantize_weight,
)
from executorch.extension.llm.export.quant.recipe import QuantConfig


class TestConvertLinear(unittest.TestCase):
    def test_int4_wraps_exportable(self):
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        module = nn.Linear(128, 64, bias=False)
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        assign_one(module, "weight", to_exportable("weight", w))

        self.assertIsInstance(module.weight.data, ExportableInt4Tensor)
        self.assertEqual(module.weight.shape, torch.Size([64, 128]))
        self.assertFalse(module.weight.requires_grad)

    def test_int8_passes_through(self):
        from torchao.quantization import IntxUnpackedToInt8Tensor

        module = nn.Linear(128, 64, bias=False)
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        self.assertIsInstance(w, IntxUnpackedToInt8Tensor)
        assign_one(module, "weight", to_exportable("weight", w))

        self.assertIsInstance(module.weight.data, IntxUnpackedToInt8Tensor)
        self.assertEqual(module.weight.shape, torch.Size([64, 128]))

    def test_int8_coarse_passes_through(self):
        """A coarse group_size passes through unchanged.

        Regrouping to a legal group_size happens in the backend pattern
        handlers at export time, so conversion leaves block_size untouched.
        """
        torch.manual_seed(0)
        weight = torch.randn(64, 256, dtype=torch.bfloat16)
        config = QuantConfig(bits=8, group_size=256, symmetric=True, method="min_max")
        w = quantize_weight(weight, config)
        before = dequantize_weight(w, torch.float32)

        module = nn.Linear(256, 64, bias=False)
        assign_one(module, "weight", to_exportable("weight", w))

        self.assertEqual(module.weight.data.block_size, (1, 256))
        after = dequantize_weight(module.weight.data, torch.float32)
        self.assertTrue(
            torch.allclose(before, after, atol=1e-5),
            f"max diff: {(before - after).abs().max():.6g}",
        )


class TestConvertLinearGroupSize16(unittest.TestCase):
    """Converting group_size=16 weights (GGUF Q6_K) preserves semantics."""

    def _make_gs16_tensor(self, N=64, K=128):
        from torchao.quantization import IntxUnpackedToInt8Tensor

        return IntxUnpackedToInt8Tensor(
            qdata=torch.randint(-32, 31, (N, K), dtype=torch.int8),
            scale=torch.randn(N, K // 16, dtype=torch.bfloat16),
            zero_point=torch.zeros(N, K // 16, dtype=torch.int8),
            target_dtype=torch.int8,
            block_size=(1, 16),
            dtype=torch.bfloat16,
            activation_quantization=None,
        )

    def test_dequant_preserves_values(self):
        """Conversion preserves the dequantized weight values."""
        w = self._make_gs16_tensor(64, 128)
        before = dequantize_weight(w, torch.float32)

        module = nn.Linear(128, 64, bias=False)
        assign_one(module, "weight", to_exportable("weight", w))
        after = dequantize_weight(module.weight.data, torch.float32)

        self.assertTrue(
            torch.allclose(before, after, atol=1e-5),
            f"max diff: {(before - after).abs().max():.6g}",
        )

    def test_forward_produces_valid_output(self):
        """Converted gs=16 weight produces finite output in a linear forward."""
        w = self._make_gs16_tensor(64, 128)
        module = nn.Linear(128, 64, bias=False)
        assign_one(module, "weight", to_exportable("weight", w))

        x = torch.randn(1, 128, dtype=torch.bfloat16)
        out = torch.nn.functional.linear(x, module.weight.data.dequantize())
        self.assertEqual(out.shape, torch.Size([1, 64]))
        self.assertFalse(torch.isnan(out).any())


class TestConvertEmbedding(unittest.TestCase):
    def test_compatible_passes_through(self):
        module = nn.Embedding(100, 64)
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(100, 64, dtype=torch.bfloat16), config)
        assign_one(module, "weight", to_exportable("weight", w))
        self.assertEqual(module.weight.shape, torch.Size([100, 64]))

    def test_per_axis_passes_through(self):
        module = nn.Embedding(50, 256)
        config = QuantConfig(bits=8, group_size=256, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(50, 256, dtype=torch.bfloat16), config)
        assign_one(module, "weight", to_exportable("weight", w))
        self.assertEqual(module.weight.shape, torch.Size([50, 256]))
        # Regrouping happens in the backend handlers at export time, not here.
        self.assertEqual(module.weight.data.block_size, (1, 256))

    def test_int4_wraps_exportable(self):
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        module = nn.Embedding(100, 64)
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(100, 64, dtype=torch.bfloat16), config)
        assign_one(module, "weight", to_exportable("weight", w))
        self.assertIsInstance(module.weight.data, ExportableInt4Tensor)
        self.assertEqual(module.weight.shape, torch.Size([100, 64]))


class TestConvertIntx4SpaceVsExportable(unittest.TestCase):
    """to_exportable leaves 4-bit intx as-is; to_default repacks to int4."""

    def _make_intx4(self, N=64, K=128, gs=32):
        from torchao.quantization import IntxUnpackedToInt8Tensor

        return IntxUnpackedToInt8Tensor(
            qdata=torch.randint(-8, 7, (N, K), dtype=torch.int8),
            scale=torch.randn(N, K // gs, dtype=torch.bfloat16),
            zero_point=torch.zeros(N, K // gs, dtype=torch.int8),
            target_dtype=torch.int4,
            block_size=(1, gs),
            dtype=torch.bfloat16,
            activation_quantization=None,
        )

    def test_exportable_leaves_intx4_unchanged(self):
        from torchao.quantization import IntxUnpackedToInt8Tensor

        w = self._make_intx4()
        module = nn.Linear(128, 64, bias=False)
        assign_one(module, "weight", to_exportable("weight", w))
        self.assertIsInstance(module.weight.data, IntxUnpackedToInt8Tensor)

    def test_space_repacks_intx4(self):
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        w = self._make_intx4()
        module = nn.Linear(128, 64, bias=False)
        assign_one(module, "weight", to_default("weight", w))
        self.assertIsInstance(module.weight.data, ExportableInt4Tensor)
        self.assertEqual(module.weight.shape, torch.Size([64, 128]))

    def test_space_convert_via_assign_state_dict(self):
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        w = self._make_intx4()
        with torch.device("meta"):
            model = nn.ModuleDict({"q_proj": nn.Linear(128, 64, bias=False)})
        assign_state_dict(model, {"q_proj.weight": w}, convert=to_default)
        self.assertIsInstance(model.q_proj.weight.data, ExportableInt4Tensor)


class TestAssignStateDict(unittest.TestCase):
    def test_mixed_precision(self):
        q4 = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        q8 = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        w4 = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), q4)
        w8 = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), q8)

        state_dict = {
            "q_proj.weight": w4,
            "v_proj.weight": w8,
            "norm.weight": torch.randn(64, dtype=torch.bfloat16),
        }

        with torch.device("meta"):
            model = nn.ModuleDict(
                {
                    "q_proj": nn.Linear(128, 64, bias=False),
                    "v_proj": nn.Linear(128, 64, bias=False),
                    "norm": nn.LayerNorm(64, bias=False),
                }
            )
        assign_state_dict(model, state_dict)

        self.assertEqual(model.q_proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model.v_proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model.norm.weight.shape, torch.Size([64]))

    def test_default_convert(self):
        """assign_state_dict with no convert uses to_exportable."""
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        q4 = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        w4 = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), q4)
        state_dict = {"q_proj.weight": w4}

        with torch.device("meta"):
            model = nn.ModuleDict({"q_proj": nn.Linear(128, 64, bias=False)})
        assign_state_dict(model, state_dict)

        self.assertIsInstance(model.q_proj.weight.data, ExportableInt4Tensor)
        self.assertEqual(model.q_proj.weight.shape, torch.Size([64, 128]))

    def test_missing_weight_detected(self):
        q4 = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        w4 = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), q4)

        with torch.device("meta"):
            model = nn.ModuleDict(
                {
                    "a": nn.Linear(128, 64, bias=False),
                    "b": nn.Linear(128, 64, bias=False),
                }
            )
        with self.assertRaises(RuntimeError) as ctx:
            assign_state_dict(model, {"a.weight": w4})
        self.assertIn("b.weight", str(ctx.exception))


class TestFuseAlongOutput(unittest.TestCase):
    """fuse_along_output concatenates weights along the output-channel dim."""

    @staticmethod
    def _int4(rows, cols=128, group_size=32):
        config = QuantConfig(
            bits=4, group_size=group_size, symmetric=False, method="min_max"
        )
        w = torch.randn(rows, cols, dtype=torch.bfloat16)
        return to_default("w", quantize_weight(w, config))

    @staticmethod
    def _intx8(rows, cols=128, group_size=32):
        config = QuantConfig(
            bits=8, group_size=group_size, symmetric=False, method="min_max"
        )
        return quantize_weight(torch.randn(rows, cols, dtype=torch.bfloat16), config)

    @staticmethod
    def _gguf(rows):
        from executorch.extension.llm.export.gguf import (
            _Q4_K_BLOCK_BYTES,
            ExportableGGUFTensor,
        )

        raw = torch.randint(0, 256, (rows, _Q4_K_BLOCK_BYTES), dtype=torch.uint8)
        return ExportableGGUFTensor(raw, "q4_k", torch.bfloat16)

    def test_plain_tensor(self):
        a = torch.randn(8, 16)
        b = torch.randn(24, 16)
        fused = fuse_along_output([a, b])
        self.assertEqual(fused.shape, torch.Size([32, 16]))
        self.assertTrue(torch.equal(fused, torch.cat([a, b], dim=0)))

    def test_single_tensor_returned_unchanged(self):
        a = self._int4(8)
        self.assertIs(fuse_along_output([a]), a)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            fuse_along_output([])

    def test_int4_fuses_exactly(self):
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        a = self._int4(8)
        b = self._int4(24)
        fused = fuse_along_output([a, b])

        self.assertIsInstance(fused, ExportableInt4Tensor)
        self.assertEqual(fused.shape, torch.Size([32, 128]))
        self.assertEqual(fused.group_size, a.group_size)
        self.assertEqual(fused.orig_dtype, a.orig_dtype)
        # qdata is N-major (dim 0); scale/zero_point are transposed (N on dim 1).
        self.assertTrue(torch.equal(fused.qdata, torch.cat([a.qdata, b.qdata], dim=0)))
        self.assertTrue(torch.equal(fused.scale, torch.cat([a.scale, b.scale], dim=1)))
        self.assertTrue(
            torch.equal(
                fused.zero_point, torch.cat([a.zero_point, b.zero_point], dim=1)
            )
        )
        ref = torch.cat([dequantize_weight(a), dequantize_weight(b)], dim=0)
        self.assertTrue(torch.allclose(dequantize_weight(fused), ref))

    def test_intx_int8_fuses_exactly(self):
        from torchao.quantization import IntxUnpackedToInt8Tensor

        a = self._intx8(8)
        b = self._intx8(24)
        fused = fuse_along_output([a, b])

        self.assertIsInstance(fused, IntxUnpackedToInt8Tensor)
        self.assertEqual(fused.shape, torch.Size([32, 128]))
        self.assertEqual(list(fused.block_size), list(a.block_size))
        self.assertEqual(fused.target_dtype, a.target_dtype)
        # Every packed field is N-major (dim 0) for IntxUnpackedToInt8Tensor.
        self.assertTrue(torch.equal(fused.qdata, torch.cat([a.qdata, b.qdata], dim=0)))
        self.assertTrue(torch.equal(fused.scale, torch.cat([a.scale, b.scale], dim=0)))
        ref = torch.cat([dequantize_weight(a), dequantize_weight(b)], dim=0)
        self.assertTrue(torch.allclose(dequantize_weight(fused), ref))

    def test_gguf_fuses_exactly(self):
        from executorch.extension.llm.export.gguf import ExportableGGUFTensor

        a = self._gguf(2)
        b = self._gguf(3)
        fused = fuse_along_output([a, b])

        self.assertIsInstance(fused, ExportableGGUFTensor)
        self.assertEqual(fused.shape, torch.Size([5, 256]))
        self.assertEqual(fused.ggml_type, "q4_k")
        self.assertEqual(fused.orig_dtype, torch.bfloat16)
        # Each row is an independent super-block, so exact fusion == cat of raw
        # bytes (random bytes decode to NaN scales, so don't compare dequant).
        self.assertTrue(torch.equal(fused.raw, torch.cat([a.raw, b.raw], dim=0)))

    def test_type_mismatch_raises(self):
        with self.assertRaises(TypeError):
            fuse_along_output([self._int4(8), self._intx8(8)])

    def test_plain_quantized_mix_raises(self):
        with self.assertRaises(TypeError):
            fuse_along_output([torch.randn(8, 128), self._int4(8)])

    def test_mismatched_quant_param_raises(self):
        a = self._int4(8, group_size=32)
        b = self._int4(8, group_size=64)
        with self.assertRaises(ValueError):
            fuse_along_output([a, b])


if __name__ == "__main__":
    unittest.main()
