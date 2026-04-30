#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Functional correctness tests for INT4 matmul and dequant Triton kernels.

Tests both int4_matmul (fused W4A16 GEMM) and dequant_w4_to_bf16 (weight
dequantization) against eager PyTorch references. Uses 0.01 absolute
tolerance to account for INT4 quantization noise and bf16 rounding.

Usage:
  python -m pytest backends/cuda/tests/test_int4_matmul.py -v
"""

import unittest

import torch

from executorch.backends.cuda.triton.kernels.int4_matmul import (
    dequant_w4_to_bf16,
    int4_matmul,
    int4_matvec,
)

ATOL = 0.01
DEVICE = "cuda"


def _quantize_simple(w_bf16, group_size):
    """Quantize [N, K] bf16 weight to simple packed INT4 + per-group scales.

    Returns:
        w_packed: [N, K//2] int8 — two INT4 values per byte
        w_scale:  [N, K//group_size] bf16 — symmetric scales
        w_ref:    [N, K] bf16 — dequantized reference matching kernel's computation
    """
    N, K = w_bf16.shape
    w = w_bf16.float()
    w_grouped = w.reshape(N, K // group_size, group_size)
    scale = w_grouped.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-10)
    int_data = (w_grouped / scale).round().clamp(-8, 7).to(torch.int8)
    # Kernel dequant: (uint4 - 8) * scale = int_data * scale
    scale_bf16 = scale.to(torch.bfloat16)
    w_ref = ((int_data.float()) * scale_bf16.float()).reshape(N, K).to(torch.bfloat16)
    scale_bf16 = scale_bf16.reshape(N, K // group_size)
    int_data = int_data.reshape(N, K)
    uint4 = (int_data + 8).to(torch.int16)
    packed = (uint4[:, 0::2] | (uint4[:, 1::2] << 4)).to(torch.int8)
    return packed.to(DEVICE), scale_bf16.to(DEVICE), w_ref.to(DEVICE)


def _eager_int4_matmul(x, w_ref):
    """Reference matmul: x @ w_ref.T in float32, cast to bf16."""
    return (x.float() @ w_ref.float().T).to(torch.bfloat16)


class TestDequantW4ToBf16(unittest.TestCase):
    """Tests for dequant_w4_to_bf16 Triton kernel."""

    def _run_dequant(self, N, K, group_size):
        torch.manual_seed(42)
        w = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
        packed, scale, w_ref = _quantize_simple(w, group_size)

        out = dequant_w4_to_bf16(packed, scale, group_size)

        self.assertEqual(out.shape, (N, K))
        self.assertEqual(out.dtype, torch.bfloat16)
        max_err = (out.float() - w_ref.float()).abs().max().item()
        self.assertLess(
            max_err, ATOL, f"dequant [{N}x{K}] gs={group_size}: max_err={max_err}"
        )

    def test_square(self):
        self._run_dequant(256, 256, 32)

    def test_tall(self):
        self._run_dequant(2048, 256, 32)

    def test_wide(self):
        self._run_dequant(256, 2048, 128)

    def test_production_qkv(self):
        self._run_dequant(2048, 2048, 128)

    def test_production_shared_expert(self):
        self._run_dequant(1024, 2048, 128)

    def test_group_size_32(self):
        self._run_dequant(512, 512, 32)

    def test_group_size_128(self):
        self._run_dequant(512, 2048, 128)

    def test_non_power_of_two_N(self):
        self._run_dequant(12352, 2048, 128)

    def test_small(self):
        self._run_dequant(16, 64, 32)


class TestInt4Matmul(unittest.TestCase):
    """Tests for int4_matmul Triton kernel (fused W4A16 GEMM)."""

    def _run_matmul(self, M, N, K, group_size):
        torch.manual_seed(42)
        w = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
        packed, scale, w_ref = _quantize_simple(w, group_size)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)

        out = int4_matmul(x, packed, scale, group_size)
        ref = _eager_int4_matmul(x, w_ref)

        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=0.01),
            f"int4_matmul M={M} [{N}x{K}] gs={group_size}: "
            f"max_abs_err={(out.float() - ref.float()).abs().max().item():.4f}, "
            f"max_rel_err={((out.float() - ref.float()).abs() / ref.float().abs().clamp(min=1e-6)).max().item():.4f}",
        )

    # --- Decode (M=1) ---
    def test_decode_square(self):
        self._run_matmul(1, 256, 256, 32)

    def test_decode_qkv(self):
        self._run_matmul(1, 2048, 2048, 128)

    def test_decode_kv_proj(self):
        self._run_matmul(1, 256, 2048, 128)

    def test_decode_shared_expert(self):
        self._run_matmul(1, 1024, 2048, 128)

    def test_decode_large_N(self):
        self._run_matmul(1, 12352, 2048, 128)

    # --- Small prefill ---
    def test_prefill_4(self):
        self._run_matmul(4, 2048, 2048, 128)

    def test_prefill_16(self):
        self._run_matmul(16, 2048, 2048, 128)

    def test_prefill_64(self):
        self._run_matmul(64, 2048, 2048, 128)

    # --- Large prefill ---
    def test_prefill_256(self):
        self._run_matmul(256, 2048, 2048, 128)

    def test_prefill_1024(self):
        self._run_matmul(1024, 2048, 2048, 128)

    def test_prefill_4095(self):
        self._run_matmul(4095, 2048, 2048, 128)

    # --- Edge cases ---
    def test_group_size_32(self):
        self._run_matmul(4, 512, 512, 32)

    def test_non_power_of_two_M(self):
        self._run_matmul(7, 256, 256, 32)

    def test_non_power_of_two_N(self):
        self._run_matmul(4, 12352, 2048, 128)

    def test_small(self):
        self._run_matmul(1, 16, 64, 32)


class TestInt4Matvec(unittest.TestCase):
    """Tests for int4_matvec Triton kernel (M=1 decode)."""

    def _run_matvec(self, N, K, group_size):
        torch.manual_seed(42)
        w = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
        packed, scale, w_ref = _quantize_simple(w, group_size)
        x = torch.randn(K, dtype=torch.bfloat16, device=DEVICE)

        out = int4_matvec(x.unsqueeze(0), packed, scale, group_size)
        ref = int4_matmul(x.unsqueeze(0), packed, scale, group_size)

        self.assertEqual(out.shape, (1, N))
        self.assertEqual(out.dtype, torch.bfloat16)
        # atol=1.0 for large accumulation across K, rtol=0.01 for relative
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=1.0, rtol=0.01),
            f"int4_matvec [{N}x{K}] gs={group_size}: "
            f"max_err={(out.float() - ref.float()).abs().max().item():.4f}, "
            f"max_rel={((out.float()-ref.float()).abs()/(ref.float().abs().clamp(min=0.1))).max().item():.4f}",
        )

    def test_qkv_proj(self):
        self._run_matvec(2048, 2048, 128)

    def test_kv_proj(self):
        self._run_matvec(256, 2048, 128)

    def test_shared_expert(self):
        self._run_matvec(1024, 2048, 128)

    def test_large_N(self):
        self._run_matvec(12352, 2048, 128)

    def test_group_size_32(self):
        self._run_matvec(512, 512, 32)

    def test_small(self):
        self._run_matvec(16, 64, 32)

    def test_matches_int4_matmul(self):
        """Matvec output matches int4_matmul at M=1."""
        torch.manual_seed(42)
        N, K, gs = 2048, 2048, 128
        w = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
        packed, scale, _ = _quantize_simple(w, gs)
        x = torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE)

        out_mv = int4_matvec(x, packed, scale, gs)
        out_mm = int4_matmul(x, packed, scale, gs)

        self.assertTrue(
            torch.allclose(out_mv.float(), out_mm.float(), atol=1.0, rtol=0.01),
            f"matvec vs matmul: max_err={(out_mv.float() - out_mm.float()).abs().max().item():.4f}",
        )


class TestDequantThenMatmul(unittest.TestCase):
    """Tests that dequant + F.linear matches int4_matmul (both paths should agree)."""

    def _run(self, M, N, K, group_size):
        torch.manual_seed(42)
        w = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
        packed, scale, w_ref = _quantize_simple(w, group_size)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)

        # Path A: fused int4_matmul
        out_fused = int4_matmul(x, packed, scale, group_size)

        # Path B: dequant + F.linear
        w_bf16 = dequant_w4_to_bf16(packed, scale, group_size)
        out_dequant = torch.nn.functional.linear(x, w_bf16)

        self.assertTrue(
            torch.allclose(
                out_fused.float(), out_dequant.float(), atol=ATOL, rtol=0.01
            ),
            f"fused vs dequant M={M} [{N}x{K}]: "
            f"max_abs_err={(out_fused.float() - out_dequant.float()).abs().max().item():.4f}",
        )

    def test_decode(self):
        self._run(1, 2048, 2048, 128)

    def test_prefill_short(self):
        self._run(64, 2048, 2048, 128)

    def test_prefill_long(self):
        self._run(1024, 2048, 2048, 128)

    def test_large_N(self):
        self._run(4, 12352, 2048, 128)


if __name__ == "__main__":
    unittest.main()
