#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Functional correctness tests for INT4 matmul and dequant Triton kernels.

Tests against precision-matched quantized references:
  - dequant_w4_to_bf16: bitwise exact vs pure-Python dequant
  - int4_matmul: vs cuBLAS bf16 GEMM on bf16-dequanted weights (both paths
    truncate dequant to bf16 before matmul, so error is just tiling order)
  - int4_matvec: vs f32 matmul on f32-dequanted weights (matvec keeps dequant
    in f32 throughout, matching the f32 reference)

Usage:
  python -m pytest backends/cuda/tests/test_int4_matmul.py -v
"""

import unittest

import torch
import torch.nn.functional as F

from executorch.backends.cuda.triton.kernels.int4_matmul import (
    dequant_w4_to_bf16,
    int4_matmul,
    int4_matvec,
)

DEVICE = "cuda"


def _pack_int4(int_data):
    """Pack [N, K] int8 values in [-8, 7] to [N, K//2] int8 (two nibbles per byte)."""
    uint4 = (int_data + 8).to(torch.int16)
    return (uint4[:, 0::2] | (uint4[:, 1::2] << 4)).to(torch.int8)


def _quantize_and_pack(N, K, group_size, device=DEVICE):
    """Create random INT4-quantized weights in simple packed format."""
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    w_grouped = w.float().reshape(N, K // group_size, group_size)
    scale = w_grouped.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-10)
    int_data = (w_grouped / scale).round().clamp(-8, 7).to(torch.int8)
    scale_bf16 = scale.to(torch.bfloat16).reshape(N, K // group_size)
    return _pack_int4(int_data.reshape(N, K)).to(device), scale_bf16.to(device)


def _python_dequant(packed, scale, group_size, output_dtype=torch.bfloat16):
    """Pure-Python INT4 dequant — no Triton, serves as ground truth.

    output_dtype=bf16 matches int4_matmul's inline dequant (truncates to bf16).
    output_dtype=f32 matches int4_matvec's inline dequant (keeps full precision).
    """
    N, K_half = packed.shape
    K = K_half * 2
    raw = packed.to(torch.uint8)
    lo = (raw & 0xF).to(torch.int8) - 8
    hi = (raw >> 4).to(torch.int8) - 8
    w_int4 = torch.stack([lo, hi], dim=-1).reshape(N, K)
    scale_expanded = scale.float().repeat_interleave(group_size, dim=1)
    w_f32 = w_int4.float() * scale_expanded
    return w_f32 if output_dtype == torch.float32 else w_f32.to(output_dtype)


# ---------------------------------------------------------------------------
# Directed byte-pattern fixtures
# ---------------------------------------------------------------------------


def _make_endpoint_weights(N, K, group_size, device=DEVICE):
    """Weights at INT4 endpoints (-8, 7) to pin nibble order and scale application."""
    int_data = torch.empty(N, K, dtype=torch.int8, device=device)
    int_data[:, 0::2] = -8
    int_data[:, 1::2] = 7
    scale = torch.ones(N, K // group_size, dtype=torch.bfloat16, device=device) * 0.5
    return _pack_int4(int_data), scale


def _make_group_boundary_weights(N, K, group_size, device=DEVICE):
    """Alternating +7/-8 within groups with distinct per-group scales."""
    int_data = torch.empty(N, K, dtype=torch.int8, device=device)
    for g in range(K // group_size):
        start = g * group_size
        int_data[:, start : start + group_size // 2] = 7
        int_data[:, start + group_size // 2 : start + group_size] = -8
    num_groups = K // group_size
    scale_vals = torch.arange(1, num_groups + 1, device=device).float() * 0.1
    scale = scale_vals.unsqueeze(0).expand(N, -1).to(torch.bfloat16)
    return _pack_int4(int_data), scale


class TestDequantW4ToBf16(unittest.TestCase):
    """Tests for dequant_w4_to_bf16 Triton kernel.

    Reference is pure-Python dequant. Both perform identical element-wise
    math (unpack uint4, subtract 8, multiply scale in f32, cast to bf16),
    so output should be bitwise exact.
    """

    def _run_dequant(self, N, K, group_size):
        torch.manual_seed(42)
        packed, scale = _quantize_and_pack(N, K, group_size)
        ref = _python_dequant(packed, scale, group_size)

        out = dequant_w4_to_bf16(packed, scale, group_size)

        self.assertEqual(out.shape, (N, K))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(out, ref),
            f"dequant [{N}x{K}] gs={group_size}: "
            f"max_err={(out.float() - ref.float()).abs().max().item()}",
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

    # --- Tail-K: K not divisible by BLOCK_K ---
    def test_tail_k_160(self):
        self._run_dequant(64, 160, 32)

    def test_tail_k_192(self):
        self._run_dequant(128, 192, 32)

    def test_tail_k_320(self):
        self._run_dequant(256, 320, 32)

    # --- Directed fixtures ---
    def test_endpoint_values(self):
        N, K, gs = 64, 256, 32
        packed, scale = _make_endpoint_weights(N, K, gs)
        ref = _python_dequant(packed, scale, gs)
        out = dequant_w4_to_bf16(packed, scale, gs)
        self.assertTrue(torch.equal(out, ref), "endpoint dequant mismatch")
        self.assertTrue((out[:, 0::2] == -4.0).all(), "low nibble -8 * 0.5 != -4.0")
        self.assertTrue((out[:, 1::2] == 3.5).all(), "high nibble 7 * 0.5 != 3.5")

    def test_group_boundary_scales(self):
        N, K, gs = 32, 256, 32
        packed, scale = _make_group_boundary_weights(N, K, gs)
        ref = _python_dequant(packed, scale, gs)
        out = dequant_w4_to_bf16(packed, scale, gs)
        self.assertTrue(torch.equal(out, ref), "group boundary dequant mismatch")

    # --- Non-contiguous stride ---
    def test_non_contiguous_packed(self):
        N, K, gs = 128, 256, 32
        packed_big, scale_big = _quantize_and_pack(N * 2, K, gs)
        packed = packed_big[::2]  # stride(0) doubled
        scale = scale_big[::2]
        self.assertFalse(packed.is_contiguous())
        ref = _python_dequant(packed.contiguous(), scale.contiguous(), gs)
        out = dequant_w4_to_bf16(packed, scale, gs)
        self.assertTrue(torch.equal(out, ref), "non-contiguous dequant mismatch")


class TestInt4Matmul(unittest.TestCase):
    """Tests for int4_matmul Triton kernel (fused W4A16 GEMM).

    int4_matmul truncates dequanted weights to bf16 before tl.dot, so the
    reference uses bf16-dequanted weights with cuBLAS bf16 GEMM. Both paths
    do bf16*bf16→f32 accumulation; error is from tiling/reduction order only.
    Effective per-element threshold is atol + rtol * |ref|.
    """

    ATOL = 0.01
    RTOL = 0.01

    def _run_matmul(self, M, N, K, group_size):
        torch.manual_seed(42)
        packed, scale = _quantize_and_pack(N, K, group_size)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)

        out = int4_matmul(x, packed, scale, group_size)
        w_bf16 = _python_dequant(packed, scale, group_size)
        ref = F.linear(x, w_bf16)

        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)
        max_abs = (out.float() - ref.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL),
            f"int4_matmul M={M} [{N}x{K}] gs={group_size}: "
            f"max_abs_err={max_abs:.6f}",
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

    # --- Tail-K: K not divisible by BLOCK_K ---
    def test_tail_k_160(self):
        self._run_matmul(4, 64, 160, 32)

    def test_tail_k_192(self):
        self._run_matmul(1, 128, 192, 32)

    def test_tail_k_320(self):
        self._run_matmul(16, 256, 320, 32)

    # --- Directed fixtures ---
    def test_endpoint_values(self):
        N, K, gs = 64, 256, 32
        packed, scale = _make_endpoint_weights(N, K, gs)
        x = torch.ones(1, K, dtype=torch.bfloat16, device=DEVICE)
        w_bf16 = _python_dequant(packed, scale, gs)
        ref = F.linear(x, w_bf16)
        out = int4_matmul(x, packed, scale, gs)
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL),
            f"endpoint matmul: max_err={(out.float()-ref.float()).abs().max().item():.6f}",
        )

    def test_group_boundary_scales(self):
        N, K, gs = 32, 256, 32
        packed, scale = _make_group_boundary_weights(N, K, gs)
        x = torch.randn(4, K, dtype=torch.bfloat16, device=DEVICE)
        w_bf16 = _python_dequant(packed, scale, gs)
        ref = F.linear(x, w_bf16)
        out = int4_matmul(x, packed, scale, gs)
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL),
            f"group boundary matmul: max_err={(out.float()-ref.float()).abs().max().item():.6f}",
        )

    # --- Non-contiguous stride ---
    def test_non_contiguous_x(self):
        torch.manual_seed(42)
        packed, scale = _quantize_and_pack(256, 256, 32)
        x_big = torch.randn(8, 256, dtype=torch.bfloat16, device=DEVICE)
        x = x_big[::2]  # stride(0) doubled, non-contiguous
        self.assertFalse(x.is_contiguous())
        ref = F.linear(x, _python_dequant(packed, scale, 32))
        out = int4_matmul(x, packed, scale, 32)
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL),
            f"non-contiguous x: max_err={(out.float()-ref.float()).abs().max().item():.6f}",
        )

    # --- API contract ---
    def test_rejects_wrong_dtype(self):
        packed, scale = _quantize_and_pack(64, 64, 32)
        x = torch.randn(1, 64, dtype=torch.float32, device=DEVICE)
        with self.assertRaises(AssertionError):
            int4_matmul(x, packed, scale, 32)

    def test_rejects_shape_mismatch(self):
        packed, scale = _quantize_and_pack(64, 128, 32)
        x = torch.randn(1, 64, dtype=torch.bfloat16, device=DEVICE)  # K mismatch
        with self.assertRaises(AssertionError):
            int4_matmul(x, packed, scale, 32)


class TestInt4Matvec(unittest.TestCase):
    """Tests for int4_matvec Triton kernel (M=1 decode).

    int4_matvec keeps dequanted weights in f32 (no bf16 truncation), so the
    reference uses f32-dequanted weights with f32 matmul. Both paths perform
    the same f32 arithmetic; error is from reduction order only.
    """

    ATOL = 0.01
    RTOL = 0.01

    def _run_matvec(self, N, K, group_size):
        torch.manual_seed(42)
        packed, scale = _quantize_and_pack(N, K, group_size)
        x = torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE)

        out = int4_matvec(x, packed, scale, group_size)
        w_f32 = _python_dequant(packed, scale, group_size, output_dtype=torch.float32)
        ref = (x.float() @ w_f32.T).to(torch.bfloat16)

        self.assertEqual(out.shape, (1, N))
        self.assertEqual(out.dtype, torch.bfloat16)
        max_abs = (out.float() - ref.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL),
            f"int4_matvec [{N}x{K}] gs={group_size}: "
            f"max_abs_err={max_abs:.6f}",
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

    # --- Tail-K ---
    def test_tail_k_160(self):
        self._run_matvec(64, 160, 32)

    def test_tail_k_192(self):
        self._run_matvec(128, 192, 32)

    # --- Directed fixtures ---
    def test_endpoint_values(self):
        N, K, gs = 64, 256, 32
        packed, scale = _make_endpoint_weights(N, K, gs)
        x = torch.ones(1, K, dtype=torch.bfloat16, device=DEVICE)
        w_f32 = _python_dequant(packed, scale, gs, output_dtype=torch.float32)
        ref = (x.float() @ w_f32.T).to(torch.bfloat16)
        out = int4_matvec(x, packed, scale, gs)
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=self.ATOL, rtol=self.RTOL),
            f"endpoint matvec: max_err={(out.float()-ref.float()).abs().max().item():.6f}",
        )

    # --- API contract ---
    def test_rejects_M_not_1(self):
        packed, scale = _quantize_and_pack(64, 64, 32)
        x = torch.randn(4, 64, dtype=torch.bfloat16, device=DEVICE)
        with self.assertRaises(AssertionError):
            int4_matvec(x, packed, scale, 32)

    def test_rejects_wrong_dtype(self):
        packed, scale = _quantize_and_pack(64, 64, 32)
        x = torch.randn(1, 64, dtype=torch.float32, device=DEVICE)
        with self.assertRaises(AssertionError):
            int4_matvec(x, packed, scale, 32)

    def test_rejects_shape_mismatch(self):
        packed, scale = _quantize_and_pack(64, 128, 32)
        x = torch.randn(1, 64, dtype=torch.bfloat16, device=DEVICE)
        with self.assertRaises(AssertionError):
            int4_matvec(x, packed, scale, 32)


class TestDequantThenMatmul(unittest.TestCase):
    """Tests that dequant + F.linear matches int4_matmul.

    Both use bf16-dequanted weights with bf16→f32 accumulation. dequant is
    bitwise exact (tested above), so error is Triton tl.dot vs cuBLAS only.
    Replaces the former cross-kernel cosine test with stricter parity
   .
    """

    ATOL = 0.01
    RTOL = 0.01

    def _run(self, M, N, K, group_size):
        torch.manual_seed(42)
        packed, scale = _quantize_and_pack(N, K, group_size)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)

        out_fused = int4_matmul(x, packed, scale, group_size)
        w_bf16 = dequant_w4_to_bf16(packed, scale, group_size)
        out_dequant = F.linear(x, w_bf16)

        max_abs = (out_fused.float() - out_dequant.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(
                out_fused.float(), out_dequant.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"fused vs dequant M={M} [{N}x{K}]: max_abs_err={max_abs:.6f}",
        )

    def test_decode(self):
        self._run(1, 2048, 2048, 128)

    def test_prefill_short(self):
        self._run(64, 2048, 2048, 128)

    def test_prefill_long(self):
        self._run(1024, 2048, 2048, 128)

    def test_large_N(self):
        self._run(4, 12352, 2048, 128)

    # --- Tail-K ---
    def test_tail_k(self):
        self._run(4, 64, 160, 32)

    # --- Directed fixtures ---
    def test_endpoint_values(self):
        N, K, gs = 64, 256, 32
        packed, scale = _make_endpoint_weights(N, K, gs)
        x = torch.randn(4, K, dtype=torch.bfloat16, device=DEVICE)
        out_fused = int4_matmul(x, packed, scale, gs)
        w_bf16 = dequant_w4_to_bf16(packed, scale, gs)
        out_dq = F.linear(x, w_bf16)
        self.assertTrue(
            torch.allclose(out_fused.float(), out_dq.float(), atol=self.ATOL, rtol=self.RTOL),
            f"endpoint fused vs dequant: max_err={(out_fused.float()-out_dq.float()).abs().max().item():.6f}",
        )


class TestDequantContract(unittest.TestCase):
    """API contract tests for dequant_w4_to_bf16."""

    def test_rejects_wrong_packed_dtype(self):
        packed = torch.zeros(8, 16, dtype=torch.float32, device=DEVICE)
        scale = torch.ones(8, 1, dtype=torch.bfloat16, device=DEVICE)
        with self.assertRaises(AssertionError):
            dequant_w4_to_bf16(packed, scale, 32)

    def test_rejects_wrong_scale_dtype(self):
        packed = torch.zeros(8, 16, dtype=torch.int8, device=DEVICE)
        scale = torch.ones(8, 1, dtype=torch.float32, device=DEVICE)
        with self.assertRaises(AssertionError):
            dequant_w4_to_bf16(packed, scale, 32)

    def test_rejects_shape_mismatch(self):
        packed = torch.zeros(8, 16, dtype=torch.int8, device=DEVICE)
        scale = torch.ones(4, 1, dtype=torch.bfloat16, device=DEVICE)  # N mismatch
        with self.assertRaises(AssertionError):
            dequant_w4_to_bf16(packed, scale, 32)

    def test_rejects_1d(self):
        packed = torch.zeros(16, dtype=torch.int8, device=DEVICE)
        scale = torch.ones(1, dtype=torch.bfloat16, device=DEVICE)
        with self.assertRaises((AssertionError, RuntimeError)):
            dequant_w4_to_bf16(packed, scale, 32)


if __name__ == "__main__":
    unittest.main()
