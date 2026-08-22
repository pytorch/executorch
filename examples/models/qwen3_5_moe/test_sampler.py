# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the standalone ``sample`` function in
``examples/models/qwen3_5_moe/sampler.py``.

``temperature`` is a runtime scalar tensor so the same exported graph can
be re-driven with different sampling configurations without re-export.

NOTE: top-k / top-p tests are intentionally omitted — that support is
deferred to a follow-up PR.

Usage:
    python -m pytest examples/models/qwen3_5_moe/test_sampler.py -v
"""

import unittest

import torch

from executorch.examples.models.qwen3_5_moe.sampler import sample


def _temp(value: float = 1.0) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32)


class TestSampler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # No-op path: when temperature is None the function returns the
    # input logits unchanged.
    # ------------------------------------------------------------------
    def test_temperature_none_returns_logits(self):
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
    # Sampling with temperature matches the inline Gumbel-max sampler
    # bit-for-bit.
    # ------------------------------------------------------------------
    def test_temperature_matches_legacy_gumbel(self):
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
    # Runtime control: changing temperature between calls produces
    # different draws without re-creating the graph.
    # ------------------------------------------------------------------
    def test_temperature_runtime_controllable(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 64) * 4.0
        argmax_id = int(logits.argmax(dim=-1).item())

        torch.manual_seed(7)
        cold = int(sample(logits, temperature=_temp(1e-4)).item())
        torch.manual_seed(7)
        hot = int(sample(logits, temperature=_temp(5.0)).item())

        # Cold sampling should hit argmax; hot sampling is unconstrained
        # but is still a valid token id.
        self.assertEqual(cold, argmax_id)
        self.assertGreaterEqual(hot, 0)
        self.assertLess(hot, logits.size(-1))

    # ------------------------------------------------------------------
    # CUDA smoke test.
    # ------------------------------------------------------------------
    def test_runs_on_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        logits = torch.randn(2, 64, device="cuda")
        temperature = torch.tensor([0.9], dtype=torch.float32, device="cuda")
        out = sample(logits, temperature=temperature)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
