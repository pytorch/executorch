# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/gguf.py — Q4_K and Q6_K unpacking.

Tests verify the API contract: dequantized canonical weights match the
original GGUF dequantization formula. Uses synthetic blocks — no GGUF
file required.
"""

import struct
import unittest

import numpy as np
import torch

try:
    from gguf import GGMLQuantizationType

    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False

if _HAS_GGUF:
    from executorch.examples.models.gemma4_31b.quant.gguf import unpack_gguf_tensor

from executorch.examples.models.gemma4_31b.quant.quantize import dequantize_weight
from executorch.examples.models.gemma4_31b.quant.serialize import deserialize, serialize


def _make_q4_k_block(d, dmin, sub_scales, sub_mins, qvals):
    """Build one Q4_K block (144 bytes) from components."""
    buf = bytearray(144)
    struct.pack_into("<e", buf, 0, d)
    struct.pack_into("<e", buf, 2, dmin)
    scales_bytes = bytearray(12)
    for j in range(4):
        scales_bytes[j] = sub_scales[j] & 0x3F
        scales_bytes[j + 4] = sub_mins[j] & 0x3F
    for j in range(4, 8):
        scales_bytes[j + 4] = (sub_scales[j] & 0xF) | ((sub_mins[j] & 0xF) << 4)
        scales_bytes[j - 4] |= (sub_scales[j] >> 4) << 6
        scales_bytes[j] |= (sub_mins[j] >> 4) << 6
    buf[4:16] = scales_bytes
    # GGUF Q4_K nibble order: 32 lows then 32 highs per sub-block pair
    for g in range(4):
        for i in range(32):
            lo_val = qvals[g * 64 + i]
            hi_val = qvals[g * 64 + 32 + i]
            buf[16 + g * 32 + i] = (lo_val & 0xF) | ((hi_val & 0xF) << 4)
    return buf


def _make_q6_k_block(d, scales_16, qvals_256):
    """Build one Q6_K block (210 bytes) from components.

    ggml processes 128 values at a time. For each 128-value half:
      ql: 64 bytes (two groups of 32, low/high nibbles)
      qh: 32 bytes (2 bits each for 4 sub-positions)
    The qvals_256 array is in output order (position 0..255).
    """
    buf = bytearray(210)
    # First half (positions 0..127): ql bytes 0..63, qh bytes 0..31
    for i in range(32):
        buf[i] = (qvals_256[i] & 0x0F) | ((qvals_256[i + 64] & 0x0F) << 4)
    for i in range(32):
        buf[32 + i] = (qvals_256[i + 32] & 0x0F) | ((qvals_256[i + 96] & 0x0F) << 4)
    for i in range(32):
        h0 = (qvals_256[i] >> 4) & 0x03
        h1 = (qvals_256[i + 32] >> 4) & 0x03
        h2 = (qvals_256[i + 64] >> 4) & 0x03
        h3 = (qvals_256[i + 96] >> 4) & 0x03
        buf[128 + i] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6)
    # Second half (positions 128..255): ql bytes 64..127, qh bytes 32..63
    for i in range(32):
        buf[64 + i] = (qvals_256[i + 128] & 0x0F) | ((qvals_256[i + 192] & 0x0F) << 4)
    for i in range(32):
        buf[96 + i] = (qvals_256[i + 160] & 0x0F) | ((qvals_256[i + 224] & 0x0F) << 4)
    for i in range(32):
        h0 = (qvals_256[i + 128] >> 4) & 0x03
        h1 = (qvals_256[i + 160] >> 4) & 0x03
        h2 = (qvals_256[i + 192] >> 4) & 0x03
        h3 = (qvals_256[i + 224] >> 4) & 0x03
        buf[160 + i] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6)
    # Scales and d
    for i in range(16):
        buf[192 + i] = scales_16[i] & 0xFF
    struct.pack_into("<e", buf, 208, d)
    return buf


def _q4_k_reference_dequant(d, dmin, sub_scales, sub_mins, qvals):
    """Reference Q4_K dequantization from the GGUF spec."""
    result = []
    for j in range(8):
        for i in range(32):
            q = qvals[j * 32 + i] % 16
            result.append(d * sub_scales[j] * q - dmin * sub_mins[j])
    return result


def _q6_k_reference_dequant(d, scales_16, qvals_256):
    """Reference Q6_K dequantization from the GGUF spec."""
    result = []
    for j in range(16):
        for i in range(16):
            q_unsigned = qvals_256[j * 16 + i]
            q_signed = q_unsigned - 32
            result.append(d * scales_16[j] * q_signed)
    return result


@unittest.skipUnless(_HAS_GGUF, "gguf package not installed")
class TestQ4KDequant(unittest.TestCase):
    def test_dequant_matches_reference(self):
        """Canonical dequant reproduces the GGUF Q4_K formula across all sub-blocks."""
        d, dmin = 0.5, 0.25
        sub_scales = [3, 7, 1, 15, 20, 10, 31, 5]
        sub_mins = [1, 2, 0, 4, 8, 3, 12, 6]
        qvals = [(i * 7 + 3) % 16 for i in range(256)]

        block = _make_q4_k_block(d, dmin, sub_scales, sub_mins, qvals)
        data = np.frombuffer(bytes(block), dtype=np.uint8).reshape(1, 144)
        cw = unpack_gguf_tensor(data, GGMLQuantizationType.Q4_K, [1, 256])

        actual = dequantize_weight(cw)[0]
        expected = torch.tensor(
            _q4_k_reference_dequant(d, dmin, sub_scales, sub_mins, qvals)
        )

        self.assertTrue(
            torch.allclose(actual, expected, atol=0.01),
            f"Max diff: {(actual - expected).abs().max():.4f}",
        )

    def test_zero_scale_produces_zero(self):
        """Scale=0 produces zero dequantized values (not dmin*min)."""
        block = _make_q4_k_block(0.0, 1.0, [0] * 8, [1] * 8, [7] * 256)
        data = np.frombuffer(bytes(block), dtype=np.uint8).reshape(1, 144)
        cw = unpack_gguf_tensor(data, GGMLQuantizationType.Q4_K, [1, 256])
        dequant = dequantize_weight(cw)
        self.assertFalse(torch.isnan(dequant).any())
        self.assertFalse(torch.isinf(dequant).any())
        # When scale=0, dequant must be 0 regardless of min/zero values.
        # Regression: previously zero_std was set to eff_min instead of 0,
        # causing nonzero dequant when scale=0 and min!=0.
        self.assertTrue((dequant == 0).all())


@unittest.skipUnless(_HAS_GGUF, "gguf package not installed")
class TestQ6KDequant(unittest.TestCase):
    def test_dequant_matches_reference(self):
        """Canonical dequant reproduces the GGUF Q6_K formula."""
        d = 0.5
        scales_16 = [i + 1 for i in range(16)]
        qvals = [(i * 3 + 5) % 64 for i in range(256)]

        block = _make_q6_k_block(d, scales_16, qvals)
        data = np.frombuffer(bytes(block), dtype=np.uint8).reshape(1, 210)
        cw = unpack_gguf_tensor(data, GGMLQuantizationType.Q6_K, [1, 256])

        actual = dequantize_weight(cw)[0]
        expected = torch.tensor(_q6_k_reference_dequant(d, scales_16, qvals))

        self.assertTrue(
            torch.allclose(actual, expected, atol=0.01),
            f"Max diff: {(actual - expected).abs().max():.4f}",
        )


@unittest.skipUnless(_HAS_GGUF, "gguf package not installed")
class TestGgufSerializeRoundtrip(unittest.TestCase):
    def test_q4_k_survives_serialize_roundtrip(self):
        """unpack → serialize → deserialize → dequant matches original."""
        d, dmin = 0.5, 0.25
        sub_scales = [3, 7, 1, 15, 20, 10, 31, 5]
        sub_mins = [1, 2, 0, 4, 8, 3, 12, 6]
        qvals = [(i * 7 + 3) % 16 for i in range(256)]

        block = _make_q4_k_block(d, dmin, sub_scales, sub_mins, qvals)
        data = np.frombuffer(bytes(block), dtype=np.uint8).reshape(1, 144)
        cw = unpack_gguf_tensor(data, GGMLQuantizationType.Q4_K, [1, 256])

        dequant_before = dequantize_weight(cw)

        tensors, header = serialize({"w": cw}, {})
        q_loaded, _ = deserialize(tensors, header)
        dequant_after = dequantize_weight(q_loaded["w"])

        self.assertTrue(
            torch.allclose(dequant_before, dequant_after, atol=0.01),
            f"Max diff: {(dequant_before - dequant_after).abs().max():.6f}",
        )

    def test_q6_k_survives_serialize_roundtrip(self):
        """unpack → serialize → deserialize → dequant matches original."""
        d = 0.5
        scales_16 = [i + 1 for i in range(16)]
        qvals = [(i * 3 + 5) % 64 for i in range(256)]

        block = _make_q6_k_block(d, scales_16, qvals)
        data = np.frombuffer(bytes(block), dtype=np.uint8).reshape(1, 210)
        cw = unpack_gguf_tensor(data, GGMLQuantizationType.Q6_K, [1, 256])

        dequant_before = dequantize_weight(cw)

        tensors, header = serialize({"w": cw}, {})
        q_loaded, _ = deserialize(tensors, header)
        dequant_after = dequantize_weight(q_loaded["w"])

        self.assertTrue(
            torch.allclose(dequant_before, dequant_after, atol=0.01),
            f"Max diff: {(dequant_before - dequant_after).abs().max():.6f}",
        )


@unittest.skipUnless(_HAS_GGUF, "gguf package not installed")
class TestUnpackGgufTensor(unittest.TestCase):
    """Tests for the public ``unpack_gguf_tensor`` API."""

    def test_f32_returns_tensor(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = unpack_gguf_tensor(data, GGMLQuantizationType.F32, [4])
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            unpack_gguf_tensor(
                np.zeros(10, dtype=np.uint8), GGMLQuantizationType.Q5_K, [1, 10]
            )


if __name__ == "__main__":
    unittest.main()
