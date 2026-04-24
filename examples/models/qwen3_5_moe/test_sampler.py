# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the standalone ``sample`` function in
``examples/models/qwen3_5_moe/sampler.py``.

Sampling parameters (``temperature``, ``top_k``, ``top_p``) are runtime
scalar tensors so the same exported graph can be re-driven with different
sampling configurations without re-export.

Usage:
    python -m pytest examples/models/qwen3_5_moe/test_sampler.py -v
"""

import unittest

import torch

from executorch.examples.models.qwen3_5_moe.sampler import sample


def _temp(value: float = 1.0) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32)


def _topk(value: int) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int64)


def _topp(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


class TestSampler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # No-op path: when temperature, top_k and top_p are all None the
    # function returns the input logits unchanged.
    # ------------------------------------------------------------------
    def test_all_none_returns_logits(self):
        logits = torch.randn(2, 8)
        out = sample(logits)
        self.assertIs(out, logits)
        self.assertEqual(out.shape, (2, 8))

    # ------------------------------------------------------------------
    # Output shape / dtype contract when sampling is enabled.
    # ------------------------------------------------------------------
    def test_output_shape_and_dtype(self):
        logits = torch.randn(3, 17)
        out = sample(logits, temperature=_temp(1.0))
        self.assertEqual(out.shape, (3, 1))
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out < logits.size(-1)))

    # ------------------------------------------------------------------
    # Sampling with temperature only matches the inline Gumbel-max sampler
    # bit-for-bit.
    # ------------------------------------------------------------------
    def test_temperature_only_matches_legacy_gumbel(self):
        logits = torch.randn(2, 32)
        temperature = _temp(0.8)

        torch.manual_seed(123)
        scaled = logits / temperature.clamp(min=1e-6)
        noise = torch.rand_like(scaled)
        gumbel = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        expected = (scaled + gumbel).argmax(dim=-1, keepdim=True).float()

        torch.manual_seed(123)
        actual = sample(logits, temperature=temperature)
        self.assertTrue(torch.equal(actual, expected))

    # ------------------------------------------------------------------
    # top_k=1 forces every sampled token to be the argmax (only one
    # survivor in the masked distribution).
    # ------------------------------------------------------------------
    def test_top_k_one_is_argmax(self):
        logits = torch.randn(4, 11)
        expected = logits.argmax(dim=-1, keepdim=True).float()
        for seed in range(5):
            torch.manual_seed(seed)
            out = sample(logits, temperature=_temp(0.7), top_k=_topk(1))
            self.assertTrue(torch.equal(out, expected))

    # ------------------------------------------------------------------
    # top_k restricts the support set: sampled IDs must lie in topk.
    # ------------------------------------------------------------------
    def test_top_k_restricts_support(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 100) * 5.0
        k = 5
        topk_ids = set(torch.topk(logits, k=k, dim=-1).indices[0].tolist())

        for trial in range(50):
            torch.manual_seed(trial)
            tok = int(sample(logits, temperature=_temp(1.0), top_k=_topk(k)).item())
            self.assertIn(
                tok, topk_ids, f"trial {trial}: token {tok} not in top-{k} set"
            )

    # ------------------------------------------------------------------
    # top_p restricts the support set to the nucleus.
    # ------------------------------------------------------------------
    def test_top_p_restricts_to_nucleus(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 50) * 4.0
        top_p = 0.5

        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        keep = cum_probs <= top_p
        keep[..., 0] = True
        nucleus_ids = set(sorted_idx[0, keep[0]].tolist())

        for trial in range(50):
            torch.manual_seed(trial + 100)
            tok = int(sample(logits, temperature=_temp(1.0), top_p=_topp(top_p)).item())
            self.assertIn(
                tok,
                nucleus_ids,
                f"trial {trial}: token {tok} not in nucleus {nucleus_ids}",
            )

    # ------------------------------------------------------------------
    # Combined top_k + top_p: result lies in the top-k intersection.
    # ------------------------------------------------------------------
    def test_top_k_and_top_p_combined(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 64) * 3.0
        k = 8
        top_p = 0.7
        topk_ids = set(torch.topk(logits, k=k, dim=-1).indices[0].tolist())

        for trial in range(30):
            torch.manual_seed(trial + 200)
            tok = int(
                sample(
                    logits,
                    temperature=_temp(0.9),
                    top_k=_topk(k),
                    top_p=_topp(top_p),
                ).item()
            )
            self.assertIn(tok, topk_ids)

    # ------------------------------------------------------------------
    # Top-k only (without temperature) still samples — confirms that any
    # one of the sampling args is enough to enable the sampling path.
    # ------------------------------------------------------------------
    def test_top_k_only_enables_sampling(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 32) * 3.0
        out = sample(logits, top_k=_topk(4))
        self.assertEqual(out.shape, (1, 1))
        self.assertEqual(out.dtype, torch.float32)

    # ------------------------------------------------------------------
    # top_k=V (vocab size) is a no-op for the top-k mask: equivalent to
    # not specifying top_k at all (with the same RNG seed).
    # ------------------------------------------------------------------
    def test_top_k_full_vocab_is_noop(self):
        logits = torch.randn(2, 16)
        torch.manual_seed(42)
        no_filter = sample(logits, temperature=_temp(1.0))
        torch.manual_seed(42)
        full_k = sample(logits, temperature=_temp(1.0), top_k=_topk(logits.size(-1)))
        self.assertTrue(torch.equal(no_filter, full_k))

    # ------------------------------------------------------------------
    # top_p=1.0 keeps every token: behaves like no top-p filter.
    # ------------------------------------------------------------------
    def test_top_p_one_is_noop(self):
        logits = torch.randn(2, 16)
        torch.manual_seed(7)
        no_filter = sample(logits, temperature=_temp(0.5))
        torch.manual_seed(7)
        full_p = sample(logits, temperature=_temp(0.5), top_p=_topp(1.0))
        self.assertTrue(torch.equal(no_filter, full_p))

    # ------------------------------------------------------------------
    # Low temperature → near-greedy sampling.
    # ------------------------------------------------------------------
    def test_low_temperature_is_near_greedy(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 32) * 2.0
        argmax_id = int(logits.argmax(dim=-1).item())

        agree = 0
        trials = 100
        for trial in range(trials):
            torch.manual_seed(trial + 500)
            tok = int(sample(logits, temperature=_temp(1e-4)).item())
            if tok == argmax_id:
                agree += 1
        self.assertGreaterEqual(agree, int(trials * 0.9))

    # ------------------------------------------------------------------
    # Empirical distribution check: top-1 should be the most frequent.
    # ------------------------------------------------------------------
    def test_distribution_peaks_at_argmax(self):
        torch.manual_seed(0)
        logits = torch.tensor([[3.0, 1.0, 0.5, -1.0, 0.0]])
        argmax_id = int(logits.argmax(dim=-1).item())

        counts = torch.zeros(logits.size(-1), dtype=torch.long)
        trials = 2000
        for trial in range(trials):
            torch.manual_seed(trial)
            tok = int(sample(logits, temperature=_temp(1.0)).item())
            counts[tok] += 1
        self.assertEqual(int(counts.argmax().item()), argmax_id)

    # ------------------------------------------------------------------
    # Runtime control: changing top_k between calls produces different
    # support sets without re-exporting (smoke test for the runtime-
    # tensor contract).
    # ------------------------------------------------------------------
    def test_top_k_runtime_controllable(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 64) * 4.0
        # Sweep k and confirm sampled tokens always lie in top-k.
        for k in (1, 3, 8, 32):
            topk_ids = set(torch.topk(logits, k=k, dim=-1).indices[0].tolist())
            for trial in range(20):
                torch.manual_seed(trial)
                tok = int(sample(logits, temperature=_temp(1.0), top_k=_topk(k)).item())
                self.assertIn(tok, topk_ids, f"k={k} trial={trial}")

    # ------------------------------------------------------------------
    # CUDA smoke test.
    # ------------------------------------------------------------------
    def test_runs_on_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        logits = torch.randn(2, 64, device="cuda")
        temperature = torch.tensor([0.9], dtype=torch.float32, device="cuda")
        top_k = torch.tensor(4, dtype=torch.int64, device="cuda")
        top_p = torch.tensor(0.95, dtype=torch.float32, device="cuda")
        out = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
