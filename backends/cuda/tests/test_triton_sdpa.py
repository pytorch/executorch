# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Comprehensive tests for the Triton SDPA kernel.

Tests MHA, GQA, MQA with various head dims, sequence lengths, causal/non-causal,
and bool masks. Reference outputs are computed using torch SDPA with expanded KV
heads (for GQA/MQA) in float32 for numerical stability.

Test parametrization adapted from FlashAttention (tests/cute/test_flash_attn.py).
"""

import itertools
import unittest

import torch
import torch.nn.functional as F


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        raise unittest.SkipTest("BF16 not supported on this GPU")


def _import_sdpa():
    from executorch.backends.cuda.triton.kernels.sdpa import sdpa

    return sdpa


def _reference_sdpa(q, k, v, attn_mask=None, is_causal=False, scale=None):
    """Compute reference SDPA in float32 with expanded KV heads for GQA.

    Adapted from FlashAttention's testing.py: expand KV heads via
    repeat_interleave, upcast to float32, use torch SDPA.
    """
    H_q = q.shape[1]
    H_kv = k.shape[1]
    num_groups = H_q // H_kv

    # Expand KV heads for GQA/MQA
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

    # Expand mask head dim if needed
    if attn_mask is not None and attn_mask.shape[1] == 1 and H_q > 1:
        attn_mask = attn_mask.expand(-1, H_q, -1, -1)

    # Upcast to float32 for reference accuracy
    return F.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
    )


def _max_abs_error(out, ref):
    return (out.float() - ref.float()).abs().max().item()


# bf16 kernel vs fp32 reference tolerance.
# The benchmark cross-validates backends at 1e-2; tests use the same bar.
MAX_ABS_TOL = 1e-2


# ---------------------------------------------------------------------------
# Test configurations adapted from FlashAttention
# ---------------------------------------------------------------------------

# Head dimensions: power-of-2 and non-power-of-2
HEAD_DIMS_POW2 = [64, 128, 256]
HEAD_DIMS_NON_POW2 = [80, 96]

# Sequence length pairs (seqlen_q, seqlen_kv) adapted from FlashAttention
# Note: Lk must be >= BLOCK_N (32) to avoid unmasked zero-padding in K
# that dilutes softmax. This is a pre-existing kernel limitation for very
# short KV sequences.
SEQLEN_PAIRS_BASIC = [
    (1, 64),
    (1, 128),
    (1, 512),
    (4, 128),
    (64, 64),
    (64, 128),
    (128, 128),
    (128, 256),
    (256, 256),
]

# GQA configurations: (H_q, H_kv, label)
GQA_CONFIGS = [
    (4, 4, "mha"),  # MHA: 1:1
    (6, 3, "gqa_2x"),  # GQA: 2 Q heads per KV head
    (8, 2, "gqa_4x"),  # GQA: 4 Q heads per KV head
    (16, 2, "gqa_8x"),  # GQA: 8 Q heads per KV head (Qwen 3.5 MoE config)
    (6, 1, "mqa"),  # MQA: all Q heads share 1 KV head
]


class TestTritonSdpa(unittest.TestCase):
    """Test Triton SDPA kernel correctness against PyTorch reference."""

    @classmethod
    def setUpClass(cls):
        _skip_if_no_cuda()
        cls.sdpa = _import_sdpa()

    # ------------------------------------------------------------------
    # MHA tests (no GQA, backwards compatibility)
    # ------------------------------------------------------------------

    def test_mha_basic(self):
        """MHA with various seqlens, pow2 head dims, no mask."""
        for D, (Lq, Lk) in itertools.product(HEAD_DIMS_POW2, SEQLEN_PAIRS_BASIC):
            if Lq > Lk:
                continue  # skip invalid causal configs
            with self.subTest(D=D, Lq=Lq, Lk=Lk):
                B, H = 2, 4
                torch.manual_seed(42)
                q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")

                out = self.sdpa(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertFalse(torch.isnan(out).any(), "NaN in output")
                self.assertLess(
                    _max_abs_error(out, ref), MAX_ABS_TOL, f"D={D} Lq={Lq} Lk={Lk}"
                )

    def test_mha_causal(self):
        """MHA with causal masking."""
        for D in HEAD_DIMS_POW2:
            for L in [64, 128, 256]:
                with self.subTest(D=D, L=L):
                    B, H = 2, 4
                    torch.manual_seed(42)
                    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda")
                    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda")

                    out = self.sdpa(q, k, v, is_causal=True)
                    ref = _reference_sdpa(q, k, v, is_causal=True)

                    self.assertFalse(torch.isnan(out).any())
                    self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_mha_bool_mask(self):
        """MHA with explicit bool attention mask."""
        B, H, D = 1, 4, 64
        for Lq, Lk in [(4, 128), (64, 64), (1, 256)]:
            with self.subTest(Lq=Lq, Lk=Lk):
                torch.manual_seed(42)
                q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")

                # Sparse mask: only first half of KV visible
                mask = torch.zeros(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")
                mask[:, :, :, : Lk // 2] = True

                out = self.sdpa(q, k, v, attn_mask=mask)
                ref = _reference_sdpa(q, k, v, attn_mask=mask)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_mha_non_pow2_head_dim(self):
        """MHA with non-power-of-2 head dimensions."""
        for D in HEAD_DIMS_NON_POW2:
            for Lq, Lk in [(1, 64), (4, 128), (64, 64), (128, 128)]:
                if Lq > Lk:
                    continue
                with self.subTest(D=D, Lq=Lq, Lk=Lk):
                    B, H = 1, 4
                    torch.manual_seed(42)
                    q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cuda")
                    k = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")

                    out = self.sdpa(q, k, v)
                    ref = _reference_sdpa(q, k, v)

                    self.assertFalse(torch.isnan(out).any())
                    self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_mha_non_pow2_causal(self):
        """MHA with non-pow2 head dim and causal masking."""
        for D in HEAD_DIMS_NON_POW2:
            for L in [64, 128]:
                with self.subTest(D=D, L=L):
                    B, H = 1, 4
                    torch.manual_seed(42)
                    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda")
                    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda")

                    out = self.sdpa(q, k, v, is_causal=True)
                    ref = _reference_sdpa(q, k, v, is_causal=True)

                    self.assertFalse(torch.isnan(out).any())
                    self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    # ------------------------------------------------------------------
    # GQA tests
    # ------------------------------------------------------------------

    def test_gqa_decode(self):
        """GQA decode (seqlen_q=1)."""
        for (H_q, H_kv, label), D, Lk in itertools.product(
            GQA_CONFIGS, [64, 128, 256], [64, 128, 512]
        ):
            if H_q == H_kv:
                continue  # skip MHA, tested above
            with self.subTest(label=label, D=D, Lk=Lk):
                B, Lq = 1, 1
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                out = self.sdpa(q, k, v, enable_gqa=True)
                ref = _reference_sdpa(q, k, v)

                self.assertEqual(out.shape, (B, H_q, Lq, D))
                self.assertFalse(torch.isnan(out).any())
                self.assertLess(
                    _max_abs_error(out, ref), MAX_ABS_TOL, f"{label} D={D} Lk={Lk}"
                )

    def test_gqa_decode_with_mask(self):
        """GQA decode with bool attention mask."""
        for H_q, H_kv, label in GQA_CONFIGS:
            if H_q == H_kv:
                continue
            with self.subTest(label=label):
                B, Lq, Lk, D = 1, 1, 256, 128
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                # Mask: only first 100 positions visible
                mask = torch.zeros(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")
                mask[:, :, :, :100] = True

                out = self.sdpa(q, k, v, attn_mask=mask, enable_gqa=True)
                ref = _reference_sdpa(q, k, v, attn_mask=mask)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_gqa_short_seqlen(self):
        """GQA with short seqlen_q (2-8)."""
        for Lq in [2, 4, 8]:
            for H_q, H_kv, label in [(8, 2, "gqa_4x"), (16, 2, "gqa_8x")]:
                with self.subTest(label=label, Lq=Lq):
                    B, Lk, D = 1, 256, 128
                    torch.manual_seed(42)
                    q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                    k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                    out = self.sdpa(q, k, v, enable_gqa=True)
                    ref = _reference_sdpa(q, k, v)

                    self.assertFalse(torch.isnan(out).any())
                    self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_gqa_prefill(self):
        """GQA prefill (long seqlen_q)."""
        for (H_q, H_kv, label), L in itertools.product(
            [(8, 2, "gqa_4x"), (16, 2, "gqa_8x"), (6, 1, "mqa")],
            [64, 128, 256],
        ):
            with self.subTest(label=label, L=L):
                B, D = 1, 128
                torch.manual_seed(42)
                q = torch.randn(B, H_q, L, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, L, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, L, D, dtype=torch.bfloat16, device="cuda")

                out = self.sdpa(q, k, v, is_causal=True, enable_gqa=True)
                ref = _reference_sdpa(q, k, v, is_causal=True)

                self.assertEqual(out.shape, (B, H_q, L, D))
                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_gqa_non_pow2_head_dim(self):
        """GQA with non-power-of-2 head dimensions."""
        for D in HEAD_DIMS_NON_POW2:
            for Lq, Lk in [(1, 128), (4, 200), (64, 64)]:
                with self.subTest(D=D, Lq=Lq, Lk=Lk):
                    B, H_q, H_kv = 1, 8, 2
                    torch.manual_seed(42)
                    q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                    k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                    out = self.sdpa(q, k, v, enable_gqa=True)
                    ref = _reference_sdpa(q, k, v)

                    self.assertFalse(torch.isnan(out).any())
                    self.assertLess(
                        _max_abs_error(out, ref), MAX_ABS_TOL, f"D={D} Lq={Lq} Lk={Lk}"
                    )

    def test_gqa_causal_prefill(self):
        """GQA with causal masking during prefill."""
        for H_q, H_kv, label in [(8, 2, "gqa_4x"), (6, 1, "mqa")]:
            for L in [64, 128]:
                with self.subTest(label=label, L=L):
                    B, D = 2, 128
                    torch.manual_seed(42)
                    q = torch.randn(B, H_q, L, D, dtype=torch.bfloat16, device="cuda")
                    k = torch.randn(B, H_kv, L, D, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn(B, H_kv, L, D, dtype=torch.bfloat16, device="cuda")

                    out = self.sdpa(q, k, v, is_causal=True, enable_gqa=True)
                    ref = _reference_sdpa(q, k, v, is_causal=True)

                    self.assertFalse(torch.isnan(out).any())
                    self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_gqa_causal_decode_with_mask(self):
        """GQA decode with causal-like bool mask (simulating KV cache)."""
        H_q, H_kv, D = 16, 2, 256
        for cache_len in [64, 256, 512]:
            with self.subTest(cache_len=cache_len):
                B, Lq = 1, 1
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(
                    B, H_kv, cache_len, D, dtype=torch.bfloat16, device="cuda"
                )
                v = torch.randn(
                    B, H_kv, cache_len, D, dtype=torch.bfloat16, device="cuda"
                )

                # KV cache mask: first `pos` entries are valid
                pos = cache_len * 3 // 4
                mask = torch.zeros(B, 1, Lq, cache_len, dtype=torch.bool, device="cuda")
                mask[:, :, :, :pos] = True

                out = self.sdpa(q, k, v, attn_mask=mask, enable_gqa=True)
                ref = _reference_sdpa(q, k, v, attn_mask=mask)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_gqa_batch_size(self):
        """GQA with batch_size > 1."""
        for B in [2, 4]:
            with self.subTest(B=B):
                H_q, H_kv, Lq, Lk, D = 8, 2, 1, 128, 128
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                out = self.sdpa(q, k, v, enable_gqa=True)
                ref = _reference_sdpa(q, k, v)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    # ------------------------------------------------------------------
    # Qwen 3.5 MoE configuration
    # ------------------------------------------------------------------

    def test_qwen35_moe_config(self):
        """Exact Qwen 3.5 MoE attention config: H_q=16, H_kv=2, D=256."""
        B, H_q, H_kv, D = 1, 16, 2, 256
        for Lq, Lk in [(1, 128), (1, 512), (1, 1024), (4, 512)]:
            with self.subTest(Lq=Lq, Lk=Lk):
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                # Simulate KV cache mask
                mask = torch.ones(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")

                out = self.sdpa(q, k, v, attn_mask=mask, enable_gqa=True)
                ref = _reference_sdpa(q, k, v, attn_mask=mask)

                self.assertEqual(out.shape, (B, H_q, Lq, D))
                self.assertFalse(torch.isnan(out).any())
                self.assertLess(
                    _max_abs_error(out, ref),
                    MAX_ABS_TOL,
                    f"Qwen config Lq={Lq} Lk={Lk}",
                )

    # ------------------------------------------------------------------
    # Edge cases and validation
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Output shape is always [B, H_q, L_q, D]."""
        B, D = 1, 64
        for H_q, H_kv in [(4, 4), (8, 2), (6, 1)]:
            for Lq, Lk in [(1, 64), (32, 64)]:
                with self.subTest(H_q=H_q, H_kv=H_kv, Lq=Lq, Lk=Lk):
                    q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                    k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                    enable = H_q != H_kv
                    out = self.sdpa(q, k, v, enable_gqa=enable)
                    self.assertEqual(out.shape, (B, H_q, Lq, D))
                    self.assertEqual(out.dtype, torch.bfloat16)

    def test_custom_scale(self):
        """Custom attention scale."""
        B, H, Lq, Lk, D = 1, 4, 1, 64, 128
        torch.manual_seed(42)
        q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")

        scale = 0.05
        out = self.sdpa(q, k, v, scale=scale)
        ref = _reference_sdpa(q, k, v, scale=scale)

        self.assertFalse(torch.isnan(out).any())
        self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_all_masked(self):
        """All-masked block should produce zeros, not NaN."""
        B, H, D = 1, 4, 64
        Lq, Lk = 4, 128
        torch.manual_seed(42)
        q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")

        # All-False mask: every entry is masked
        mask = torch.zeros(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")
        out = self.sdpa(q, k, v, attn_mask=mask)

        self.assertFalse(torch.isnan(out).any(), "All-masked should not NaN")
        self.assertFalse(torch.isinf(out).any(), "All-masked should not Inf")

    def test_gqa_validation_errors(self):
        """Invalid GQA configs should raise RuntimeError."""
        B, D = 1, 64

        # H_q not divisible by H_kv
        q = torch.randn(B, 5, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, 3, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, 3, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.sdpa(q, k, v, enable_gqa=True)

        # H_q != H_kv without enable_gqa
        q = torch.randn(B, 8, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, 2, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, 2, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.sdpa(q, k, v, enable_gqa=False)

    def test_per_head_mask_rejected(self):
        """Per-head masks (H>1) should be rejected since the kernel broadcasts."""
        B, H, Lq, Lk, D = 1, 4, 4, 64, 64
        q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
        mask = torch.ones(B, H, Lq, Lk, dtype=torch.bool, device="cuda")
        with self.assertRaises(RuntimeError):
            self.sdpa(q, k, v, attn_mask=mask)

    def test_gqa_all_masked_decode(self):
        """GQA decode with all-masked block should not NaN."""
        B, H_q, H_kv, Lq, Lk, D = 1, 8, 2, 1, 128, 64
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        mask = torch.zeros(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")
        out = self.sdpa(q, k, v, attn_mask=mask, enable_gqa=True)

        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_causal_lq_ne_lkv_rejected(self):
        """is_causal=True with L_q != L_kv should raise RuntimeError."""
        B, H, D = 1, 4, 64
        q = torch.randn(B, H, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, 128, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, 128, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.sdpa(q, k, v, is_causal=True)

    def test_non_pow2_no_mask(self):
        """Non-pow2 head dim without mask should work (mask_ptr=0 path)."""
        B, H, Lq, Lk, D = 1, 4, 4, 64, 40  # D=40 is not pow2
        torch.manual_seed(42)
        q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")

        out = self.sdpa(q, k, v)
        ref = _reference_sdpa(q, k, v)

        self.assertFalse(torch.isnan(out).any())
        self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)


if __name__ == "__main__":
    unittest.main()
