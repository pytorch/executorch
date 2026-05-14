# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the split-K decode SDPA kernel (sdpa_decode_splitk).

Mirrors test_triton_sdpa.py structure. Reference outputs use torch SDPA with
expanded KV heads in float32.
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


def _import_splitk():
    from executorch.backends.cuda.triton.kernels.sdpa import sdpa_decode_splitk

    return sdpa_decode_splitk


def _import_sdpa():
    from executorch.backends.cuda.triton.kernels.sdpa import sdpa

    return sdpa


def _reference_sdpa(q, k, v, attn_mask=None, scale=None):
    """Compute reference SDPA in float32 with expanded KV heads for GQA."""
    H_q = q.shape[1]
    H_kv = k.shape[1]
    num_groups = H_q // H_kv

    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

    if attn_mask is not None and attn_mask.shape[1] == 1 and H_q > 1:
        attn_mask = attn_mask.expand(-1, H_q, -1, -1)

    return F.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=attn_mask,
        scale=scale,
    )


def _max_abs_error(out, ref):
    return (out.float() - ref.float()).abs().max().item()


# bf16 kernel vs fp32 reference tolerance.
# Matches benchmark cross-validation and test_triton_sdpa.py.
MAX_ABS_TOL = 1e-2


HEAD_DIMS_POW2 = [64, 128, 256]

GQA_CONFIGS = [
    (6, 3, "gqa_2x"),
    (8, 2, "gqa_4x"),
    (16, 2, "gqa_8x"),
    (6, 1, "mqa"),
]

LK_LENGTHS = [64, 128, 512, 1024, 4096]


class TestTritonSdpaSplitK(unittest.TestCase):
    """Test split-K decode SDPA kernel correctness against PyTorch reference."""

    @classmethod
    def setUpClass(cls):
        _skip_if_no_cuda()
        cls.splitk = _import_splitk()
        cls.sdpa = _import_sdpa()

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_decode_basic(self):
        """GQA decode across head configs, head dims, and KV lengths."""
        for (H_q, H_kv, label), D, Lk in itertools.product(
            GQA_CONFIGS,
            HEAD_DIMS_POW2,
            LK_LENGTHS,
        ):
            with self.subTest(label=label, D=D, Lk=Lk):
                B, Lq = 1, 1
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                out = self.splitk(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertEqual(out.shape, (B, H_q, Lq, D))
                self.assertFalse(torch.isnan(out).any(), "NaN in output")
                self.assertLess(
                    _max_abs_error(out, ref),
                    0.05,
                    f"{label} D={D} Lk={Lk}",
                )

    def test_decode_with_mask(self):
        """Decode with bool mask (KV cache style: first N positions valid)."""
        for H_q, H_kv, label in GQA_CONFIGS:
            with self.subTest(label=label):
                B, Lq, Lk, D = 1, 1, 512, 128
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                mask = torch.zeros(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")
                mask[:, :, :, :200] = True

                out = self.splitk(q, k, v, attn_mask=mask)
                ref = _reference_sdpa(q, k, v, attn_mask=mask)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_decode_mha(self):
        """MHA (H_q==H_kv, num_groups=1) should work with split-K."""
        for D, Lk in itertools.product([64, 128], [128, 512]):
            with self.subTest(D=D, Lk=Lk):
                B, H, Lq = 1, 4, 1
                torch.manual_seed(42)
                q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H, Lk, D, dtype=torch.bfloat16, device="cuda")

                out = self.splitk(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_qwen35_config(self):
        """Exact Qwen3.5 MoE config: H_q=16, H_kv=2, D=256."""
        B, H_q, H_kv, D = 1, 16, 2, 256
        for Lk in [128, 512, 1024, 4096]:
            with self.subTest(Lk=Lk):
                Lq = 1
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                mask = torch.ones(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")

                out = self.splitk(q, k, v, attn_mask=mask)
                ref = _reference_sdpa(q, k, v, attn_mask=mask)

                self.assertEqual(out.shape, (B, H_q, Lq, D))
                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_custom_scale(self):
        """Non-default attention scale."""
        B, H_q, H_kv, Lq, Lk, D = 1, 8, 2, 1, 256, 128
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        scale = 0.05
        out = self.splitk(q, k, v, scale=scale)
        ref = _reference_sdpa(q, k, v, scale=scale)

        self.assertFalse(torch.isnan(out).any())
        self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_cross_validate_with_sdpa(self):
        """Split-K output matches tiled sdpa output for decode shapes."""
        B, H_q, H_kv, D = 1, 8, 2, 128
        for Lk in [128, 512, 1024]:
            with self.subTest(Lk=Lk):
                Lq = 1
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                mask = torch.ones(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")

                out_splitk = self.splitk(q, k, v, attn_mask=mask)
                out_tiled = self.sdpa(q, k, v, attn_mask=mask, enable_gqa=True)

                self.assertLess(
                    _max_abs_error(out_splitk, out_tiled),
                    MAX_ABS_TOL,
                    f"Split-K vs tiled mismatch at Lk={Lk}",
                )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_all_masked(self):
        """All-False mask should produce zeros, not NaN."""
        B, H_q, H_kv, Lq, Lk, D = 1, 8, 2, 1, 128, 64
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        mask = torch.zeros(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")
        out = self.splitk(q, k, v, attn_mask=mask)

        self.assertFalse(torch.isnan(out).any(), "All-masked should not NaN")
        self.assertFalse(torch.isinf(out).any(), "All-masked should not Inf")

    def test_lk_1(self):
        """Degenerate single KV position (num_splits=1)."""
        B, H_q, H_kv, Lq, Lk, D = 1, 4, 2, 1, 1, 64
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        out = self.splitk(q, k, v)
        ref = _reference_sdpa(q, k, v)

        self.assertFalse(torch.isnan(out).any())
        self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_batch_size(self):
        """Batch size > 1."""
        for B in [2, 4]:
            with self.subTest(B=B):
                H_q, H_kv, Lq, Lk, D = 8, 2, 1, 256, 128
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

                out = self.splitk(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_lq_not_1_rejected(self):
        """L_q != 1 should raise RuntimeError."""
        B, H_q, H_kv, D = 1, 8, 2, 64
        q = torch.randn(B, H_q, 4, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.splitk(q, k, v)

    def test_dropout_rejected(self):
        """dropout_p != 0 should raise RuntimeError."""
        B, H_q, H_kv, D = 1, 8, 2, 64
        q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.splitk(q, k, v, dropout_p=0.1)

    def test_is_causal_accepted(self):
        """is_causal=True is a no-op at L_q=1, should not raise."""
        B, H_q, H_kv, D = 1, 8, 2, 64
        q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        out = self.splitk(q, k, v, is_causal=True)
        self.assertEqual(out.shape, (B, H_q, 1, D))

    def test_hq_not_divisible_rejected(self):
        """H_q % H_kv != 0 should raise RuntimeError."""
        B, D = 1, 64
        q = torch.randn(B, 5, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, 3, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, 3, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.splitk(q, k, v)

    def test_non_pow2_d_rejected(self):
        """Non-power-of-2 D should raise RuntimeError."""
        B, H_q, H_kv, D = 1, 8, 2, 96
        q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.splitk(q, k, v)


if __name__ == "__main__":
    unittest.main()
