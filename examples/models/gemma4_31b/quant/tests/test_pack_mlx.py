# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/pack_mlx.py. No CUDA or MLX hardware required."""

import unittest

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.quant.pack import pack_model
from executorch.examples.models.gemma4_31b.quant.pack_mlx import (
    _mlx_group_size,
    DEFAULT_MLX_PACKERS,
    pack_for_mlx,
)
from executorch.examples.models.gemma4_31b.quant.quantize import (
    dequantize_weight,
    quantize_weight,
)
from executorch.examples.models.gemma4_31b.quant.recipe import QuantConfig


class TestPackLinearForMlx(unittest.TestCase):
    def test_int4_wraps_exportable(self):
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        module = nn.Linear(128, 64, bias=False)
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        pack_for_mlx(module, {"weight": w})

        self.assertIsInstance(module.weight.data, ExportableInt4Tensor)
        self.assertEqual(module.weight.shape, torch.Size([64, 128]))
        self.assertFalse(module.weight.requires_grad)

    def test_int8_passes_through(self):
        from torchao.quantization import IntxUnpackedToInt8Tensor

        module = nn.Linear(128, 64, bias=False)
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(64, 128, dtype=torch.bfloat16), config)
        self.assertIsInstance(w, IntxUnpackedToInt8Tensor)
        pack_for_mlx(module, {"weight": w})

        self.assertIsInstance(module.weight.data, IntxUnpackedToInt8Tensor)
        self.assertEqual(module.weight.shape, torch.Size([64, 128]))

    def test_regroup_preserves_dequant(self):
        """Linear with non-standard group_size regroups and dequantizes correctly."""
        torch.manual_seed(0)
        weight = torch.randn(64, 256, dtype=torch.bfloat16)
        config = QuantConfig(bits=8, group_size=256, symmetric=True, method="min_max")
        w = quantize_weight(weight, config)
        before = dequantize_weight(w, torch.float32)

        module = nn.Linear(256, 64, bias=False)
        pack_for_mlx(module, {"weight": w})

        self.assertEqual(module.weight.data.block_size, (1, 128))
        after = dequantize_weight(module.weight.data, torch.float32)
        self.assertTrue(
            torch.allclose(before, after, atol=1e-5),
            f"max diff: {(before - after).abs().max():.6g}",
        )


class TestMlxGroupSize(unittest.TestCase):
    def test_passthrough(self):
        for gs in (16, 32, 64, 128):
            self.assertEqual(_mlx_group_size(gs, 256), gs)

    def test_regroup_5376(self):
        self.assertEqual(_mlx_group_size(5376, 5376), 128)

    def test_regroup_256(self):
        self.assertEqual(_mlx_group_size(256, 256), 128)

    def test_rejects_indivisible(self):
        with self.assertRaises(ValueError):
            _mlx_group_size(7, 7)


class TestPackLinearGroupSize16(unittest.TestCase):
    """Packing group_size=16 weights (GGUF Q6_K) preserves semantics."""

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
        """Packing preserves the dequantized weight values."""
        w = self._make_gs16_tensor(64, 128)
        before = dequantize_weight(w, torch.float32)

        module = nn.Linear(128, 64, bias=False)
        pack_for_mlx(module, {"weight": w})
        after = dequantize_weight(module.weight.data, torch.float32)

        self.assertTrue(
            torch.allclose(before, after, atol=1e-5),
            f"max diff: {(before - after).abs().max():.6g}",
        )

    def test_forward_produces_valid_output(self):
        """Packed gs=16 weight produces finite output in a linear forward."""
        w = self._make_gs16_tensor(64, 128)
        module = nn.Linear(128, 64, bias=False)
        pack_for_mlx(module, {"weight": w})

        x = torch.randn(1, 128, dtype=torch.bfloat16)
        out = torch.nn.functional.linear(x, module.weight.data.dequantize())
        self.assertEqual(out.shape, torch.Size([1, 64]))
        self.assertFalse(torch.isnan(out).any())


class TestPackEmbeddingForMlx(unittest.TestCase):
    def test_compatible_passes_through(self):
        module = nn.Embedding(100, 64)
        config = QuantConfig(bits=8, group_size=32, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(100, 64, dtype=torch.bfloat16), config)
        pack_for_mlx(module, {"weight": w})
        self.assertEqual(module.weight.shape, torch.Size([100, 64]))

    def test_per_axis_regroups(self):
        module = nn.Embedding(50, 256)
        config = QuantConfig(bits=8, group_size=256, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(50, 256, dtype=torch.bfloat16), config)
        pack_for_mlx(module, {"weight": w})
        self.assertEqual(module.weight.shape, torch.Size([50, 256]))
        self.assertEqual(module.weight.data.block_size, (1, 128))

    def test_int4_wraps_exportable(self):
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor

        module = nn.Embedding(100, 64)
        config = QuantConfig(bits=4, group_size=32, symmetric=True, method="min_max")
        w = quantize_weight(torch.randn(100, 64, dtype=torch.bfloat16), config)
        pack_for_mlx(module, {"weight": w})
        self.assertIsInstance(module.weight.data, ExportableInt4Tensor)
        self.assertEqual(module.weight.shape, torch.Size([100, 64]))


class TestPackModelMlx(unittest.TestCase):
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
        pack_model(model, state_dict, DEFAULT_MLX_PACKERS)

        self.assertEqual(model.q_proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model.v_proj.weight.shape, torch.Size([64, 128]))
        self.assertEqual(model.norm.weight.shape, torch.Size([64]))


if __name__ == "__main__":
    unittest.main()
