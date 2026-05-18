# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TQ4 fused SDPA kernel (tq4_sdpa).

Verifies that attention over nibble-packed TQ4-compressed K/V cache
matches a reference path: decompress K/V → standard SDPA in float32.

Test structure follows test_triton_sdpa.py.
"""

import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.triton.kernels.tq4_sdpa import tq4_sdpa
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.llm.modules.turboquant import TurboQuantKVCache
from executorch.extension.llm.modules.turboquant.codebook import (
    generate_rotation_matrix,
    solve_lloyd_max,
)
from torch.export import export


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        raise unittest.SkipTest("BF16 not supported on this GPU")


def _make_codebook_and_rotation(head_dim, bits=4, seed=42):
    """Precompute TQ4 constants."""
    centroids, boundaries = solve_lloyd_max(head_dim, bits)
    rotation = generate_rotation_matrix(head_dim, seed=seed)
    return centroids, boundaries, rotation


def _compress(x, boundaries, rotation):
    """Compress (B, H, S, D) tensor to nibble-packed uint8 + fp32 norms."""
    B, H, S, D = x.shape
    flat = x.reshape(-1, D).float()

    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    normalized = flat / (norms + 1e-10)
    rotated = normalized @ rotation.float().T
    indices = torch.bucketize(rotated, boundaries.float())

    idx_u8 = indices.to(torch.uint8)
    packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]

    return packed.reshape(B, H, S, D // 2), norms.reshape(B, H, S, 1)


def _decompress(packed, norms, centroids, rotation):
    """Decompress nibble-packed uint8 + fp32 norms to float tensor."""
    B, H, S, half_D = packed.shape
    D = half_D * 2
    flat = packed.reshape(-1, half_D)
    flat_norms = norms.reshape(-1, 1)

    high = (flat >> 4).long()
    low = (flat & 0x0F).long()
    indices = torch.stack([high, low], dim=-1).reshape(-1, D)

    reconstructed = centroids.float()[indices]
    unrotated = reconstructed @ rotation.float()
    scaled = unrotated * flat_norms

    return scaled.reshape(B, H, S, D)


def _reference_tq4_sdpa(
    q, k, v, centroids, boundaries, rotation, attn_mask=None, is_causal=False
):
    """Reference: compress K/V, decompress, run standard SDPA in float32."""
    k_packed, k_norms = _compress(k, boundaries, rotation)
    v_packed, v_norms = _compress(v, boundaries, rotation)

    k_dec = _decompress(k_packed, k_norms, centroids, rotation)
    v_dec = _decompress(v_packed, v_norms, centroids, rotation)

    H_q = q.shape[1]
    H_kv = k.shape[1]
    if H_q != H_kv:
        k_dec = k_dec.repeat_interleave(H_q // H_kv, dim=1)
        v_dec = v_dec.repeat_interleave(H_q // H_kv, dim=1)

    if attn_mask is not None and attn_mask.shape[1] == 1 and H_q > 1:
        attn_mask = attn_mask.expand(-1, H_q, -1, -1)

    out = F.scaled_dot_product_attention(
        q.float(),
        k_dec.float(),
        v_dec.float(),
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
    return out, k_packed, k_norms, v_packed, v_norms


def _cosine_sim(a, b):
    return F.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


# Test configs
HEAD_DIMS = [64, 128, 256]
SEQLEN_PAIRS = [
    (1, 64),
    (1, 128),
    (4, 64),
    (64, 64),
    (128, 128),
]
GQA_CONFIGS = [
    (4, 4, "mha"),
    (4, 2, "gqa_2x"),
    (8, 2, "gqa_4x"),
    (16, 2, "gqa_8x"),
    (6, 1, "mqa"),
]


class TestTQ4Sdpa(unittest.TestCase):
    """Test TQ4 fused SDPA kernel against decompress-then-SDPA reference."""

    @classmethod
    def setUpClass(cls):
        _skip_if_no_cuda()
        cls.tq4_sdpa = tq4_sdpa

    def _run_test(
        self, B, H_q, H_kv, Lq, Lk, D, attn_mask=None, is_causal=False, min_cosine=0.95
    ):
        torch.manual_seed(42)
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids = centroids.cuda()
        boundaries = boundaries.cuda()
        rotation = rotation.cuda()

        q = torch.randn(B, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")

        ref_out, k_packed, k_norms, v_packed, v_norms = _reference_tq4_sdpa(
            q,
            k,
            v,
            centroids,
            boundaries,
            rotation,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        out = self.tq4_sdpa(
            q,
            k_packed.cuda(),
            k_norms.cuda(),
            v_packed.cuda(),
            v_norms.cuda(),
            centroids,
            rotation,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        self.assertFalse(torch.isnan(out).any(), "NaN in output")
        cos = _cosine_sim(out, ref_out)
        self.assertGreater(
            cos,
            min_cosine,
            f"Cosine {cos:.4f} < {min_cosine} "
            f"(B={B} H_q={H_q} H_kv={H_kv} Lq={Lq} Lk={Lk} D={D})",
        )

    # ------------------------------------------------------------------
    # MHA (H_q == H_kv)
    # ------------------------------------------------------------------

    def test_mha_basic(self):
        """MHA with various head dims and sequence lengths."""
        for D in HEAD_DIMS:
            for Lq, Lk in SEQLEN_PAIRS:
                if Lq > Lk:
                    continue
                with self.subTest(D=D, Lq=Lq, Lk=Lk):
                    self._run_test(1, 4, 4, Lq, Lk, D)

    def test_mha_causal(self):
        """MHA with is_causal=True."""
        for D in [64, 128, 256]:
            for L in [64, 128]:
                with self.subTest(D=D, L=L):
                    self._run_test(1, 4, 4, L, L, D, is_causal=True)

    def test_mha_causal_explicit_mask(self):
        """MHA with causal masking via explicit bool mask."""
        for D in [64, 128, 256]:
            for L in [64, 128]:
                with self.subTest(D=D, L=L):
                    mask = torch.tril(
                        torch.ones(1, 1, L, L, dtype=torch.bool, device="cuda")
                    )
                    self._run_test(1, 4, 4, L, L, D, attn_mask=mask)

    def test_mha_bool_mask(self):
        """MHA with explicit bool attention mask."""
        D = 64
        for Lq, Lk in [(1, 64), (4, 128), (1, 256)]:
            with self.subTest(Lq=Lq, Lk=Lk):
                mask = torch.zeros(1, 1, Lq, Lk, dtype=torch.bool, device="cuda")
                mask[:, :, :, : Lk // 2] = True
                self._run_test(1, 4, 4, Lq, Lk, D, attn_mask=mask)

    # ------------------------------------------------------------------
    # GQA (H_q > H_kv)
    # ------------------------------------------------------------------

    def test_gqa_decode(self):
        """GQA decode (seqlen_q=1)."""
        for H_q, H_kv, label in GQA_CONFIGS:
            if H_q == H_kv:
                continue
            for D in [64, 128, 256]:
                with self.subTest(label=label, D=D):
                    self._run_test(1, H_q, H_kv, 1, 128, D)

    def test_gqa_prefill(self):
        """GQA prefill (seqlen_q > 1) with is_causal=True."""
        for H_q, H_kv, label in GQA_CONFIGS:
            if H_q == H_kv:
                continue
            with self.subTest(label=label):
                self._run_test(1, H_q, H_kv, 64, 64, 128, is_causal=True)

    def test_gqa_8x_head_dim_256(self):
        """GQA 8:1 with head_dim=256 — matches Qwen 3.5 MoE config."""
        self._run_test(1, 16, 2, 1, 128, 256)
        L = 64
        mask = torch.tril(torch.ones(1, 1, L, L, dtype=torch.bool, device="cuda"))
        self._run_test(1, 16, 2, L, L, 256, attn_mask=mask)

    def test_gqa_with_mask(self):
        """GQA decode with explicit bool mask."""
        D, Lk = 128, 128
        mask = torch.ones(1, 1, 1, Lk, dtype=torch.bool, device="cuda")
        mask[:, :, :, Lk // 2 :] = False
        self._run_test(1, 8, 2, 1, Lk, D, attn_mask=mask)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_batch_size_2(self):
        """Batch size > 1."""
        self._run_test(2, 4, 2, 1, 64, 128)

    def test_short_kv(self):
        """Short KV sequence (32 tokens)."""
        self._run_test(1, 4, 4, 1, 32, 64)

    def test_all_masked_produces_zeros(self):
        """Fully masked Q rows produce zero output, not NaN."""
        D = 64
        torch.manual_seed(42)
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids, rotation = centroids.cuda(), rotation.cuda()

        q = torch.randn(1, 4, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 4, 64, D, dtype=torch.bfloat16, device="cuda")

        k_packed, k_norms = _compress(k, boundaries.cuda(), rotation)
        v_packed, v_norms = _compress(v, boundaries.cuda(), rotation)

        # All-False mask: every KV position is masked out
        mask = torch.zeros(1, 1, 1, 64, dtype=torch.bool, device="cuda")
        out = self.tq4_sdpa(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            centroids,
            rotation,
            mask,
        )
        self.assertFalse(torch.isnan(out).any(), "NaN in output with all-masked row")
        self.assertFalse(torch.isinf(out).any(), "Inf in output with all-masked row")
        self.assertEqual(out.abs().max().item(), 0.0)

    def test_sparse_mask_no_nan(self):
        """Sparse mask with many all-masked tile blocks produces no NaN/Inf.

        Only a few KV positions are unmasked, so most tile blocks are entirely
        masked (-inf). The softmax must not produce NaN from -inf - (-inf)
        and must not propagate it into subsequent valid blocks.
        """
        D, Lk = 64, 256
        torch.manual_seed(42)
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids, boundaries, rotation = (
            centroids.cuda(),
            boundaries.cuda(),
            rotation.cuda(),
        )

        q = torch.randn(1, 4, 4, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, Lk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 4, Lk, D, dtype=torch.bfloat16, device="cuda")

        k_packed, k_norms = _compress(k, boundaries, rotation)
        v_packed, v_norms = _compress(v, boundaries, rotation)

        # Sparse: only positions 100-103 unmasked, rest masked.
        mask = torch.zeros(1, 1, 4, Lk, dtype=torch.bool, device="cuda")
        mask[:, :, :, 100:104] = True

        out = self.tq4_sdpa(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            centroids,
            rotation,
            mask,
        )
        self.assertFalse(torch.isnan(out).any(), "NaN with sparse mask")
        self.assertFalse(torch.isinf(out).any(), "Inf with sparse mask")
        self.assertGreater(out.abs().max().item(), 0, "Output is all zeros")

    def test_float_mask_rejected(self):
        """Float attention mask raises RuntimeError."""
        D = 64
        torch.manual_seed(42)
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids, rotation = centroids.cuda(), rotation.cuda()

        q = torch.randn(1, 4, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, 64, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 4, 64, D, dtype=torch.bfloat16, device="cuda")
        k_packed, k_norms = _compress(k, boundaries.cuda(), rotation)
        v_packed, v_norms = _compress(v, boundaries.cuda(), rotation)

        float_mask = torch.zeros(1, 1, 1, 64, dtype=torch.float32, device="cuda")
        with self.assertRaises(RuntimeError):
            self.tq4_sdpa(
                q,
                k_packed,
                k_norms,
                v_packed,
                v_norms,
                centroids,
                rotation,
                float_mask,
            )

    def test_qwen35_moe_config(self):
        """Qwen 3.5 MoE: head_dim=256, GQA 16:2, decode + prefill."""
        self._run_test(1, 16, 2, 1, 256, 256)
        self._run_test(1, 16, 2, 128, 128, 256, is_causal=True)

    def test_mqa(self):
        """MQA (all Q heads share 1 KV head)."""
        for D in [64, 128]:
            with self.subTest(D=D):
                self._run_test(1, 6, 1, 1, 128, D)

    def test_gqa_short_seqlen(self):
        """GQA with short seqlen_q (2-8), exercises Pack GQA boundary."""
        for Lq in [2, 4, 8]:
            with self.subTest(Lq=Lq):
                self._run_test(1, 8, 2, Lq, 128, 128)

    def test_gqa_long_kv(self):
        """GQA decode with longer KV sequences."""
        for Lk in [512, 1024]:
            with self.subTest(Lk=Lk):
                self._run_test(1, 16, 2, 1, Lk, 128)

    def test_gqa_causal_decode_with_cache_mask(self):
        """GQA decode with KV cache mask at various fill levels."""
        H_q, H_kv, D = 16, 2, 128
        for cache_len in [64, 256, 512]:
            with self.subTest(cache_len=cache_len):
                pos = cache_len * 3 // 4
                mask = torch.zeros(1, 1, 1, cache_len, dtype=torch.bool, device="cuda")
                mask[:, :, :, :pos] = True
                self._run_test(1, H_q, H_kv, 1, cache_len, D, attn_mask=mask)

    def test_output_shape_and_dtype(self):
        """Output shape and dtype are correct for various configs."""
        for H_q, H_kv in [(4, 4), (8, 2), (6, 1)]:
            for Lq, Lk in [(1, 64), (32, 64)]:
                with self.subTest(H_q=H_q, H_kv=H_kv, Lq=Lq, Lk=Lk):
                    D = 64
                    torch.manual_seed(42)
                    centroids, boundaries, rotation = _make_codebook_and_rotation(D)
                    centroids, boundaries, rotation = (
                        centroids.cuda(),
                        boundaries.cuda(),
                        rotation.cuda(),
                    )
                    q = torch.randn(1, H_q, Lq, D, dtype=torch.bfloat16, device="cuda")
                    k = torch.randn(1, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn(1, H_kv, Lk, D, dtype=torch.bfloat16, device="cuda")
                    k_p, k_n = _compress(k, boundaries, rotation)
                    v_p, v_n = _compress(v, boundaries, rotation)
                    out = self.tq4_sdpa(
                        q,
                        k_p,
                        k_n,
                        v_p,
                        v_n,
                        centroids,
                        rotation,
                    )
                    self.assertEqual(out.shape, (1, H_q, Lq, D))
                    self.assertEqual(out.dtype, torch.bfloat16)

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_hq_not_divisible_by_hkv_rejected(self):
        """H_Q not divisible by H_KV raises RuntimeError."""
        D = 64
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids, boundaries, rotation = (
            centroids.cuda(),
            boundaries.cuda(),
            rotation.cuda(),
        )
        q = torch.randn(1, 5, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 3, 64, D, dtype=torch.bfloat16, device="cuda")
        k_p, k_n = _compress(k, boundaries, rotation)
        v_p, v_n = _compress(k, boundaries, rotation)
        with self.assertRaises(RuntimeError):
            self.tq4_sdpa(q, k_p, k_n, v_p, v_n, centroids, rotation)

    def test_causal_lq_ne_lkv_rejected(self):
        """is_causal=True with L_Q != L_KV raises RuntimeError."""
        D = 64
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids, boundaries, rotation = (
            centroids.cuda(),
            boundaries.cuda(),
            rotation.cuda(),
        )
        q = torch.randn(1, 4, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, 64, D, dtype=torch.bfloat16, device="cuda")
        k_p, k_n = _compress(k, boundaries, rotation)
        v_p, v_n = _compress(k, boundaries, rotation)
        with self.assertRaises(RuntimeError):
            self.tq4_sdpa(
                q,
                k_p,
                k_n,
                v_p,
                v_n,
                centroids,
                rotation,
                is_causal=True,
            )

    def test_non_pow2_head_dim_rejected(self):
        """Non-power-of-2 HEAD_DIM raises RuntimeError."""
        D = 80
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids, rotation = centroids.cuda(), rotation.cuda()
        q = torch.randn(1, 4, 1, D, dtype=torch.bfloat16, device="cuda")
        k_p = torch.zeros(1, 4, 64, D // 2, dtype=torch.uint8, device="cuda")
        k_n = torch.zeros(1, 4, 64, 1, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(RuntimeError):
            self.tq4_sdpa(q, k_p, k_n, k_p, k_n, centroids, rotation)

    def test_per_head_mask_rejected(self):
        """Per-head masks (H>1) should be rejected since the kernel broadcasts."""
        D = 64
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids, boundaries, rotation = (
            centroids.cuda(),
            boundaries.cuda(),
            rotation.cuda(),
        )
        q = torch.randn(1, 4, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, 64, D, dtype=torch.bfloat16, device="cuda")
        k_p, k_n = _compress(k, boundaries, rotation)
        v_p, v_n = _compress(k, boundaries, rotation)
        # H=4 instead of H=1
        mask = torch.ones(1, 4, 1, 64, dtype=torch.bool, device="cuda")
        with self.assertRaises(RuntimeError):
            self.tq4_sdpa(q, k_p, k_n, v_p, v_n, centroids, rotation, mask)

    def test_mask_shape_mismatch_rejected(self):
        """Mask with wrong B/Lq/Lkv dims raises RuntimeError."""
        D = 64
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)
        centroids, boundaries, rotation = (
            centroids.cuda(),
            boundaries.cuda(),
            rotation.cuda(),
        )
        q = torch.randn(1, 4, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 4, 64, D, dtype=torch.bfloat16, device="cuda")
        k_p, k_n = _compress(k, boundaries, rotation)
        v_p, v_n = _compress(k, boundaries, rotation)
        # Wrong Lkv: 32 instead of 64
        mask = torch.ones(1, 1, 1, 32, dtype=torch.bool, device="cuda")
        with self.assertRaises(RuntimeError):
            self.tq4_sdpa(q, k_p, k_n, v_p, v_n, centroids, rotation, mask)

    # ------------------------------------------------------------------
    # Full path: TurboQuantKVCache + tq4_sdpa
    # ------------------------------------------------------------------

    def test_kv_cache_plus_sdpa(self):
        """TurboQuantKVCache.update() -> tq4_sdpa matches reference SDPA."""
        D, H_Q, H_KV, MAX_SEQ = 128, 8, 2, 64
        torch.manual_seed(42)
        centroids, boundaries, rotation = _make_codebook_and_rotation(D)

        cache = TurboQuantKVCache(H_KV, D, MAX_SEQ).cuda()

        # Prefill 16 tokens
        k_pf = torch.randn(1, H_KV, 16, D, dtype=torch.bfloat16, device="cuda")
        v_pf = torch.randn(1, H_KV, 16, D, dtype=torch.bfloat16, device="cuda")
        pos_pf = torch.arange(16, device="cuda")
        k_packed, k_norms, v_packed, v_norms = cache.update(pos_pf, k_pf, v_pf)

        # Decode query
        q = torch.randn(1, H_Q, 1, D, dtype=torch.bfloat16, device="cuda")
        mask = torch.zeros(1, 1, 1, MAX_SEQ, dtype=torch.bool, device="cuda")
        mask[:, :, :, :16] = True

        out = self.tq4_sdpa(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            cache.centroids,
            cache.rotation,
            mask,
        )

        # Reference: use test's own decompress + standard SDPA
        k_dec = _decompress(
            k_packed[:, :, :16],
            k_norms[:, :, :16],
            centroids.cuda(),
            rotation.cuda(),
        )
        v_dec = _decompress(
            v_packed[:, :, :16],
            v_norms[:, :, :16],
            centroids.cuda(),
            rotation.cuda(),
        )
        k_dec = k_dec.repeat_interleave(H_Q // H_KV, dim=1)
        v_dec = v_dec.repeat_interleave(H_Q // H_KV, dim=1)
        ref = F.scaled_dot_product_attention(q.float(), k_dec.float(), v_dec.float())

        cos = _cosine_sim(out, ref)
        self.assertGreater(cos, 0.95, f"Cosine {cos:.4f}")
        self.assertFalse(torch.isnan(out).any())

    def test_kv_cache_decode_accumulates(self):
        """Decode tokens accumulate in cache and affect attention output."""
        D, H_Q, H_KV, MAX_SEQ = 64, 4, 2, 32
        torch.manual_seed(42)

        cache = TurboQuantKVCache(H_KV, D, MAX_SEQ).cuda()

        # Insert tokens one at a time, keep update() return values
        for i in range(8):
            k = torch.randn(1, H_KV, 1, D, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(1, H_KV, 1, D, dtype=torch.bfloat16, device="cuda")
            k_packed, k_norms, v_packed, v_norms = cache.update(
                torch.tensor([i], device="cuda"), k, v
            )

        q = torch.randn(1, H_Q, 1, D, dtype=torch.bfloat16, device="cuda")
        mask = torch.zeros(1, 1, 1, MAX_SEQ, dtype=torch.bool, device="cuda")
        mask[:, :, :, :8] = True

        out = self.tq4_sdpa(
            q,
            k_packed,
            k_norms,
            v_packed,
            v_norms,
            cache.centroids,
            cache.rotation,
            mask,
        )

        self.assertFalse(torch.isnan(out).any())
        self.assertGreater(out.abs().max().item(), 0, "Output is all zeros")

    # ------------------------------------------------------------------
    # Export through CUDA backend
    # ------------------------------------------------------------------

    def test_export_cuda(self):
        """Export tq4_sdpa through CudaPartitioner, verify .pte is produced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pte_path, _ = _export_tq4_attn(tmpdir)
            self.assertTrue(os.path.exists(pte_path))
            self.assertGreater(os.path.getsize(pte_path), 0)

    def test_e2e_cpp_runner(self):
        """Export once, run executor_runner with multiple inputs, compare."""
        if not os.path.exists(RUNNER_PATH):
            self.skipTest(
                f"executor_runner not found at {RUNNER_PATH}. "
                "Build with: cmake --build cmake-out --target executor_runner"
            )

        D, H_Q, SEQ = 128, 4, 64
        e2e_seeds = [0, 7, 42]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = os.path.join(tmpdir, "export")
            pte_path, model = _export_tq4_attn(export_dir)
            ptd_path = os.path.join(export_dir, "aoti_cuda_blob.ptd")

            for seed in e2e_seeds:
                with self.subTest(seed=seed):
                    inputs = _make_tq4_inputs(seed, H_Q, D, SEQ)

                    with torch.no_grad():
                        ref = model(*inputs)

                    run_dir = os.path.join(tmpdir, f"run_seed{seed}")
                    os.makedirs(run_dir)

                    input_files = []
                    for i, tensor in enumerate(inputs):
                        path = os.path.join(run_dir, f"{i}.bin")
                        _save_tensor(tensor, path)
                        input_files.append(path)

                    output_base = os.path.join(run_dir, "output")
                    result = _run_cpp_runner(
                        RUNNER_PATH,
                        pte_path,
                        ptd_path,
                        input_files,
                        output_base,
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        f"seed={seed}: executor_runner failed:\n{result.stderr}",
                    )

                    cpp_out = _load_output(
                        f"{output_base}-0.bin",
                        (1, H_Q, 2, D),
                        torch.bfloat16,
                    )

                    cos = _cosine_sim(cpp_out, ref.cpu())
                    self.assertGreater(
                        cos,
                        0.99,
                        f"seed={seed}: cosine {cos:.4f}",
                    )


# ---------------------------------------------------------------------------
# Export + runner helpers
# ---------------------------------------------------------------------------

EXECUTORCH_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
RUNNER_PATH = os.path.join(EXECUTORCH_ROOT, "cmake-out", "executor_runner")


class _TQ4AttnModule(torch.nn.Module):
    """Minimal module wrapping tq4_sdpa for export testing."""

    def __init__(self, head_dim, h_q, h_kv, max_seq):
        super().__init__()
        centroids, boundaries, rotation = _make_codebook_and_rotation(head_dim)
        self.register_buffer("centroids", centroids)
        self.register_buffer("rotation", rotation)

        # Pre-populate with compressed random data so outputs are non-zero
        k = torch.randn(1, h_kv, max_seq, head_dim)
        v = torch.randn(1, h_kv, max_seq, head_dim)
        k_packed, k_norms = _compress(k, boundaries, rotation)
        v_packed, v_norms = _compress(v, boundaries, rotation)
        self.register_buffer("k_packed", k_packed)
        self.register_buffer("k_norms", k_norms.to(torch.bfloat16))
        self.register_buffer("v_packed", v_packed)
        self.register_buffer("v_norms", v_norms.to(torch.bfloat16))

    def forward(self, query, attn_mask):
        return tq4_sdpa(
            query,
            self.k_packed,
            self.k_norms,
            self.v_packed,
            self.v_norms,
            self.centroids,
            self.rotation,
            attn_mask,
        )


def _export_tq4_attn(output_dir):
    """Export a _TQ4AttnModule to .pte + .ptd. Returns (pte_path, model)."""
    D, H_Q, H_KV, SEQ = 128, 4, 2, 64

    torch.manual_seed(42)
    model = _TQ4AttnModule(D, H_Q, H_KV, SEQ).to("cuda").eval()
    inputs = _make_tq4_inputs(42, H_Q, D, SEQ)

    with torch.no_grad():
        ep = export(model, inputs, strict=True)

    os.makedirs(output_dir, exist_ok=True)

    specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    et_prog = to_edge_transform_and_lower(
        ep,
        partitioner=[CudaPartitioner(specs)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    pte_path = os.path.join(output_dir, "tq4_sdpa.pte")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)

    if hasattr(et_program, "_tensor_data") and et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)

    return pte_path, model


def _make_tq4_inputs(seed, h_q, head_dim, max_seq, device="cuda"):
    torch.manual_seed(seed)
    q = torch.randn(1, h_q, 2, head_dim, dtype=torch.bfloat16, device=device)
    mask = torch.ones(1, 1, 2, max_seq, dtype=torch.bool, device=device)
    return (q, mask)


def _save_tensor(t, path):
    t_cpu = t.cpu().contiguous()
    with open(path, "wb") as f:
        f.write(bytes(t_cpu.untyped_storage()))


def _load_output(path, shape, dtype):
    data = np.fromfile(path, dtype=np.uint8)
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def _run_cpp_runner(runner_path, pte_path, ptd_path, input_files, output_base):
    cmd = [
        runner_path,
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--inputs={','.join(input_files)}",
        f"--output_file={output_base}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
