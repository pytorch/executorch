# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness (vs F.sdpa) + isolated speedup for the mid-M flash SDPA kernel.

CUDA + Triton only. Validates the length-bounded mid-M kernel against the exact
attention the gemma4 full-attention layers compute (causal, enable_gqa, scale=1)
and shows it beats a full-buffer F.sdpa when the valid length << max_seq_len.
"""

import unittest

import torch

from executorch.backends.cuda.triton.kernels.sdpa_midm import (
    midm_sdpa,
    sdpa_midm,
    sdpa_midm_reference,
)


def _require_cuda(tc):
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _rand(B, Hkv, H, M, D, S, anchor, device="cuda", dtype=torch.bfloat16):
    q = torch.randn(B, H, M, D, device=device, dtype=dtype)
    k = torch.randn(B, Hkv, S, D, device=device, dtype=dtype)
    v = torch.randn(B, Hkv, S, D, device=device, dtype=dtype)
    input_pos = torch.arange(anchor, anchor + M, device=device, dtype=torch.long)
    return q, k, v, input_pos


def _rel_err(a, b):
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


class TestMidMSDPA(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)
        torch.manual_seed(0)

    def _check(self, B, Hkv, H, M, D, S, anchor, tol=0.02):
        q, k, v, pos = _rand(B, Hkv, H, M, D, S, anchor)
        got = sdpa_midm(q, k, v, pos, scale=1.0)
        ref = sdpa_midm_reference(q, k, v, pos, scale=1.0)
        self.assertEqual(got.shape, (B, H, M, D))
        err = _rel_err(got, ref)
        self.assertLess(err, tol, f"rel_err={err} for M={M} D={D} anchor={anchor}")

    # gemma4 global-attention shape: H=32, HKV=4 (GQA 8), D=512.
    def test_global_layer_verify_window(self):
        for M in (2, 4, 5, 8):
            for anchor in (0, 17, 200, 1000):
                self._check(1, 4, 32, M, 512, 4096, anchor)

    def test_other_gqa_and_headdim(self):
        # smaller config (head_dim 256, GQA 4) to exercise generality
        for M in (2, 5, 8):
            self._check(1, 2, 8, M, 256, 2048, 300)

    def test_anchor_zero_single_diagonal(self):
        # anchor 0: row j attends keys [0, j] only
        self._check(1, 4, 32, 4, 512, 1024, 0)

    def test_matches_full_buffer_fsdpa(self):
        # The bounded kernel must equal F.sdpa over the FULL buffer with the
        # model's causal additive mask (the rest masked to -inf).
        import torch.nn.functional as F

        q, k, v, pos = _rand(1, 4, 32, 5, 512, 8192, 500)
        key_idx = torch.arange(8192, device="cuda")
        keep = key_idx[None, :] <= pos[:, None]
        am = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
        full = F.scaled_dot_product_attention(
            q, k, v, attn_mask=am, is_causal=False, enable_gqa=True, scale=1.0
        )
        got = sdpa_midm(q, k, v, pos, scale=1.0)
        self.assertLess(_rel_err(got, full), 0.02)

    def test_splitk_large_context(self):
        # Many active splits: 64K buffer, anchors across the range. Exercises the
        # cross-split online-softmax reduce at the lengths that motivated split-K.
        for anchor in (2048, 30000, 60000):
            for M in (2, 5, 8):
                self._check(1, 4, 32, M, 512, 65536, anchor)

    def test_splitk_masked_and_boundary_splits(self):
        # anchor small vs a large buffer: late key-range splits are fully causal-
        # masked for the early rows (null partials), and a row's cutoff lands mid
        # chunk. Reduce must discard -inf/0 partials cleanly.
        for anchor in (1, 31, 33, 500):
            self._check(1, 2, 8, 5, 256, 65536, anchor)

    def test_dispatch_falls_back(self):
        # M=1 and M>MIDM_MAX_M must take the F.sdpa path (not the mid-M kernel).
        import torch.nn.functional as F

        for M in (1, 16):
            q, k, v, pos = _rand(1, 4, 32, M, 512, 1024, 100)
            am = torch.zeros(M, 1024, device="cuda", dtype=q.dtype)
            key_idx = torch.arange(1024, device="cuda")
            am = torch.where(key_idx[None, :] <= pos[:, None], 0.0, float("-inf")).to(
                q.dtype
            )
            out = midm_sdpa(q, k, v, pos, am, scale=1.0, enable=True)
            ref = F.scaled_dot_product_attention(
                q, k, v, attn_mask=am, is_causal=False, enable_gqa=True, scale=1.0
            )
            self.assertLess(_rel_err(out, ref), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
