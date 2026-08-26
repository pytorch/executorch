# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the split-K decode SDPA kernel (sdpa_decode_splitk).

Mirrors test_triton_sdpa.py structure. Reference outputs use torch SDPA with
expanded KV heads in float32.
"""

import importlib
import itertools
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        raise unittest.SkipTest("BF16 not supported on this GPU")


def _import_legacy_splitk():
    from executorch.backends.cuda.triton.kernels.sdpa import sdpa_decode_splitk

    return sdpa_decode_splitk


def _import_small_query_splitk():
    from executorch.backends.cuda.triton.kernels import sdpa_small_query_splitk

    return sdpa_small_query_splitk


def _import_sdpa_module():
    return importlib.import_module("executorch.backends.cuda.triton.kernels.sdpa")


def _import_splitk_config():
    from executorch.backends.cuda.triton.kernels.sdpa import _decode_splitk_config

    return _decode_splitk_config


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
LEGACY_SPLITK_PHI = 5.0
FLOAT32_LOG_MAX = math.log(torch.finfo(torch.float32).max)
FLOAT32_LOG_MIN_SUBNORMAL = math.log(2**-149)


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
        cls.legacy_splitk = _import_legacy_splitk()
        cls.small_query_splitk = _import_small_query_splitk()
        cls.sdpa_module = _import_sdpa_module()
        cls.sdpa = cls.sdpa_module.sdpa
        cls.splitk_config = staticmethod(_import_splitk_config())

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

                out = self.legacy_splitk(q, k, v)
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

                out = self.legacy_splitk(q, k, v, attn_mask=mask)
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

                out = self.legacy_splitk(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_small_query_blocks(self):
        """Split-K supports each small-query length through the verifier size."""
        B, H_q, H_kv, Lk, D = 1, 8, 2, 512, 128
        for Lq in [2, 3, 4]:
            with self.subTest(Lq=Lq):
                torch.manual_seed(42)
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                mask = torch.ones(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")

                out = self.small_query_splitk(q, k, v, attn_mask=mask)
                ref = _reference_sdpa(q, k, v, attn_mask=mask)

                self.assertEqual(out.shape, (B, H_q, Lq, D))
                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_non_power_of_two_gqa_ratio_with_three_queries(self):
        """Padded group tiles must retain every query row."""
        B, H_q, H_kv, Lq, Lk, D = 1, 10, 2, 3, 512, 128
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        mask = torch.ones(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")

        out = self.small_query_splitk(q, k, v, attn_mask=mask)
        ref = _reference_sdpa(q, k, v, attn_mask=mask)

        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_public_dispatch_isolates_kernel_families(self):
        """Lq1 and verifier blocks must use disjoint split-K launchers."""
        B, H_q, H_kv, Lk, D = 1, 8, 2, 256, 64
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        kv_len = torch.tensor(Lk, dtype=torch.int64, device="cuda")

        for Lq, legacy_calls, small_query_calls in [(1, 1, 0), (4, 0, 1)]:
            with self.subTest(Lq=Lq):
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                mask = torch.ones(B, 1, Lq, Lk, dtype=torch.bool, device="cuda")
                with mock.patch.object(
                    self.sdpa_module, "_launch_decode_splitk"
                ) as legacy_launcher, mock.patch.object(
                    self.sdpa_module, "_launch_small_query_splitk"
                ) as small_query_launcher:
                    self.sdpa(
                        q,
                        k,
                        v,
                        attn_mask=mask,
                        enable_gqa=True,
                        kv_len=kv_len,
                    )

                self.assertEqual(legacy_launcher.call_count, legacy_calls)
                self.assertEqual(small_query_launcher.call_count, small_query_calls)

    def test_high_gqa_small_query_with_runtime_kv_len(self):
        """Exercise Lq=4, 16:1 GQA, and a bottom-right causal mask."""
        B, H_q, H_kv, Lq, Lk, D = 1, 32, 2, 4, 4096, 128
        valid_len = 4089
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        q_pos = torch.arange(valid_len - Lq, valid_len, device="cuda")
        k_pos = torch.arange(Lk, device="cuda")
        mask = (q_pos[:, None] >= k_pos[None, :])[None, None]
        kv_len = torch.tensor(valid_len, dtype=torch.int64, device="cuda")

        out = self.sdpa(
            q,
            k,
            v,
            attn_mask=mask,
            enable_gqa=True,
            kv_len=kv_len,
        )
        ref = _reference_sdpa(q, k, v, attn_mask=mask)

        self.assertFalse(torch.isnan(out).any())
        self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_runtime_kv_len_ignores_garbage_tail(self):
        """The public split-K dispatch must not read past the valid KV prefix."""
        B, H_q, H_kv, Lq, Lk, D = 1, 32, 2, 4, 131072, 128
        valid_len = 509
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        k[:, :, valid_len:] = 1000
        v[:, :, valid_len:] = 1000
        q_pos = torch.arange(valid_len - Lq, valid_len, device="cuda")
        k_pos = torch.arange(Lk, device="cuda")
        mask = (q_pos[:, None] >= k_pos[None, :])[None, None]
        kv_len = torch.tensor(valid_len, dtype=torch.int64, device="cuda")

        out = self.sdpa(
            q,
            k,
            v,
            attn_mask=mask,
            enable_gqa=True,
            kv_len=kv_len,
        )
        ref = _reference_sdpa(
            q,
            k[:, :, :valid_len],
            v[:, :, :valid_len],
            attn_mask=mask[:, :, :, :valid_len],
        )

        self.assertTrue(torch.isfinite(out).all())
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

                out = self.legacy_splitk(q, k, v, attn_mask=mask)
                ref = _reference_sdpa(q, k, v, attn_mask=mask)

                self.assertEqual(out.shape, (B, H_q, Lq, D))
                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_large_positive_logits_stable(self):
        """Large logits stay finite in both split-K kernel families."""
        B, H_q, H_kv, Lk, D = 1, 32, 8, 512, 128
        num_splits, chunk_size = self.splitk_config(Lk, B * H_kv, torch.device("cuda"))
        self.assertGreater(num_splits, 1)
        self.assertNotEqual(0 // chunk_size, 300 // chunk_size)

        k = torch.zeros(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        # Put nearly equal high scores in separate 256-token splits. The old
        # fixed-phi path overflowed both partial softmaxes, and their proximity
        # makes the result sensitive to correct cross-split rescaling.
        k[:, :, 0, :] = 3.0
        k[:, :, 300, :] = 2.96875
        torch.manual_seed(42)
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        for Lq, splitk in [
            (1, self.legacy_splitk),
            (2, self.small_query_splitk),
            (4, self.small_query_splitk),
        ]:
            with self.subTest(Lq=Lq):
                q = torch.full(
                    (B, H_q, Lq, D), 3.0, dtype=torch.bfloat16, device="cuda"
                )

                # Guard against weakening the inputs below the old kernel's
                # overflow boundary after subtracting its fixed phi.
                high_scores = torch.stack(
                    [
                        (q[0, 0, 0].float() * k[0, 0, pos].float()).sum() / D**0.5
                        for pos in (0, 300)
                    ]
                )
                self.assertTrue(
                    ((high_scores - LEGACY_SPLITK_PHI) > FLOAT32_LOG_MAX).all()
                )

                out = splitk(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertTrue(torch.isfinite(out).all())
                # The tolerance includes BF16 output rounding for this
                # concentrated two-key distribution.
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_large_negative_logits_do_not_underflow_to_zero(self):
        """Large negative logits retain their normalized weighted values."""
        B, H_q, H_kv, Lk, D = 1, 32, 8, 512, 128
        k = torch.full((B, H_kv, Lk, D), -3.0, dtype=torch.bfloat16, device="cuda")
        v = torch.ones(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        for Lq, splitk in [
            (1, self.legacy_splitk),
            (2, self.small_query_splitk),
            (4, self.small_query_splitk),
        ]:
            with self.subTest(Lq=Lq):
                q = torch.full(
                    (B, H_q, Lq, D), 3.0, dtype=torch.bfloat16, device="cuda"
                )

                # The old fixed-phi exponent was below float32's smallest
                # subnormal, so every weight became zero.
                max_score = (q[0, 0, 0].float() * k[0, 0, 0].float()).sum() / D**0.5
                self.assertLess(
                    max_score.item() - LEGACY_SPLITK_PHI,
                    FLOAT32_LOG_MIN_SUBNORMAL,
                )

                out = splitk(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertTrue(torch.isfinite(out).all())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_non_power_of_two_split_count(self):
        """Reduction masks lanes beyond the runtime split count."""
        B, H_q, H_kv, Lk, D = 1, 8, 2, 768, 128
        num_splits, _ = self.splitk_config(Lk, B * H_kv, torch.device("cuda"))
        self.assertEqual(num_splits, 6)

        torch.manual_seed(42)
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        for Lq, splitk in [
            (1, self.legacy_splitk),
            (4, self.small_query_splitk),
        ]:
            with self.subTest(Lq=Lq):
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                out = splitk(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertTrue(torch.isfinite(out).all())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_custom_scale(self):
        """Non-default attention scale."""
        B, H_q, H_kv, Lq, Lk, D = 1, 8, 2, 1, 256, 128
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        scale = 0.05
        out = self.legacy_splitk(q, k, v, scale=scale)
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

                out_splitk = self.legacy_splitk(q, k, v, attn_mask=mask)
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
        out = self.legacy_splitk(q, k, v, attn_mask=mask)

        self.assertFalse(torch.isnan(out).any(), "All-masked should not NaN")
        self.assertFalse(torch.isinf(out).any(), "All-masked should not Inf")

    def test_kv_len_overwrites_poisoned_partial_buffers(self):
        """Every valid partial slot is written, including for empty splits."""
        B, H_q, H_kv, Lk, D = 1, 8, 2, 512, 128
        valid_kv_len = 200
        torch.manual_seed(42)
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        kv_len = torch.tensor([valid_kv_len], dtype=torch.int32, device="cuda")

        real_empty = torch.empty

        for Lq, splitk in [
            (1, self.legacy_splitk),
            (4, self.small_query_splitk),
        ]:
            with self.subTest(Lq=Lq):
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                poisoned_allocations = 0

                def poisoned_empty(*args, **kwargs):
                    nonlocal poisoned_allocations
                    result = real_empty(*args, **kwargs)
                    if result.device.type == "cuda" and result.dtype == torch.float32:
                        result.fill_(float("nan"))
                        poisoned_allocations += 1
                    return result

                with mock.patch.object(torch, "empty", side_effect=poisoned_empty):
                    out = splitk(q, k, v, kv_len=kv_len)

                ref = _reference_sdpa(q, k[:, :, :valid_kv_len], v[:, :, :valid_kv_len])

                self.assertGreaterEqual(poisoned_allocations, 3)
                self.assertTrue(torch.isfinite(out).all())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    def test_lk_1(self):
        """Degenerate single KV position (num_splits=1)."""
        B, H_q, H_kv, Lq, Lk, D = 1, 4, 2, 1, 1, 64
        torch.manual_seed(42)
        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        out = self.legacy_splitk(q, k, v)
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

                out = self.legacy_splitk(q, k, v)
                ref = _reference_sdpa(q, k, v)

                self.assertFalse(torch.isnan(out).any())
                self.assertLess(_max_abs_error(out, ref), MAX_ABS_TOL)

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_legacy_lq_two_rejected(self):
        """The legacy decode op remains restricted to L_q == 1."""
        B, H_q, H_kv, D = 1, 8, 2, 64
        q = torch.randn(B, H_q, 2, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.legacy_splitk(q, k, v)

    def test_small_query_lq_one_and_five_rejected(self):
        """The small-query op accepts only L_q values 2 through 4."""
        B, H_q, H_kv, D = 1, 8, 2, 64
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        for Lq in [1, 5]:
            with self.subTest(Lq=Lq):
                q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                with self.assertRaises(RuntimeError):
                    self.small_query_splitk(q, k, v)

    def test_multi_query_implicit_causal_rejected(self):
        """Cached multi-query attention requires an explicit aligned mask."""
        B, H_q, H_kv, D = 1, 8, 2, 64
        q = torch.randn(B, H_q, 4, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.small_query_splitk(q, k, v, is_causal=True)

    def test_dropout_rejected(self):
        """dropout_p != 0 should raise RuntimeError."""
        B, H_q, H_kv, D = 1, 8, 2, 64
        q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.legacy_splitk(q, k, v, dropout_p=0.1)

    def test_is_causal_accepted(self):
        """is_causal=True is a no-op at L_q=1, should not raise."""
        B, H_q, H_kv, D = 1, 8, 2, 64
        q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        out = self.legacy_splitk(q, k, v, is_causal=True)
        self.assertEqual(out.shape, (B, H_q, 1, D))

    def test_hq_not_divisible_rejected(self):
        """H_q % H_kv != 0 should raise RuntimeError."""
        B, D = 1, 64
        q = torch.randn(B, 5, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, 3, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, 3, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.legacy_splitk(q, k, v)

    def test_non_pow2_d_rejected(self):
        """Non-power-of-2 D should raise RuntimeError."""
        B, H_q, H_kv, D = 1, 8, 2, 96
        q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, 64, D, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.legacy_splitk(q, k, v)


if __name__ == "__main__":
    unittest.main()
