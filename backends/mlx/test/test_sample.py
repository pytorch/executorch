#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export and on-device tests for mlx::sample (Gumbel-max token sampling).

The end-to-end cases run the exported program through the compiled
op_test_runner (see backends/mlx/test/README.md).

Usage:
    python -m unittest executorch.backends.mlx.test.test_sample
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional

# Registers torch.ops.mlx.sample.
import executorch.backends.mlx.custom_ops  # noqa: F401
import torch
import torch.nn as nn
from executorch.backends.mlx.llm.sampling import SamplingHead
from executorch.backends.mlx.test.test_utils import (
    export_model_to_pte,
    load_tensors_from_bin,
    run_cpp_test_runner,
    save_tensors_to_bin,
)


class _LogitsPassthrough(nn.Module):
    """Stand-in for a model returning logits [B, S, vocab]."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


class SeededSampleModel(nn.Module):
    """SamplingHead with temperature AND seed as runtime forward inputs."""

    def __init__(self):
        super().__init__()
        self.head = SamplingHead(_LogitsPassthrough())

    def forward(self, logits, temperature, seed):
        return self.head(logits, temperature=temperature, seed=seed)


class TopPSampleModel(nn.Module):
    """SamplingHead with temperature, seed, and top_p as runtime inputs."""

    def __init__(self):
        super().__init__()
        self.head = SamplingHead(_LogitsPassthrough())

    def forward(self, logits, temperature, seed, top_p):
        return self.head(logits, temperature=temperature, seed=seed, top_p=top_p)


class TopKSampleModel(nn.Module):
    """SamplingHead with temperature, seed, and top_k as runtime inputs."""

    def __init__(self):
        super().__init__()
        self.head = SamplingHead(_LogitsPassthrough())

    def forward(self, logits, temperature, seed, top_k):
        return self.head(logits, temperature=temperature, seed=seed, top_k=top_k)


def _ref_gumbel_max(logits: torch.Tensor, temperature: float, seed: int):
    """Independent Gumbel-max reference using the same torch RNG as the op."""
    gen = torch.Generator().manual_seed(seed)
    u = torch.rand(logits.shape, generator=gen)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits / temperature + gumbel, dim=-1)


def _tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two discrete distributions."""
    return 0.5 * torch.abs(p - q).sum().item()


def _sample(
    logits,
    temperature,
    seed: Optional[int],
    top_p: float = 1.0,
    top_k: Optional[int] = None,
):
    t = torch.tensor(float(temperature))
    s = None if seed is None else torch.tensor(int(seed), dtype=torch.int64)
    p = torch.tensor(float(top_p))  # 1.0 = off
    k = None if top_k is None else torch.tensor(int(top_k), dtype=torch.int64)
    return torch.ops.mlx.sample(logits, t, p, s, k)


class TestSampleOp(unittest.TestCase):
    """Eager reference behavior of mlx::sample (no export / no runtime)."""

    def test_matches_independent_gumbel_reference(self):
        # Same seed -> bit-identical token vs an independent Gumbel-max impl.
        torch.manual_seed(1)
        logits = torch.randn(8, 512)
        for seed in (0, 1, 7, 42):
            got = _sample(logits, 0.8, seed=seed)
            expected = _ref_gumbel_max(logits, 0.8, seed)
            self.assertTrue(torch.equal(got, expected), f"mismatch at seed={seed}")

    def test_distribution_matches_softmax(self):
        # Empirical token frequencies match softmax(logits / T).
        vocab = 5
        temperature = 1.0
        torch.manual_seed(0)
        base = torch.randn(vocab)
        n = 20000
        tokens = _sample(base.expand(n, vocab), temperature, seed=0)

        empirical = torch.bincount(tokens, minlength=vocab).float() / n
        target = torch.softmax(base / temperature, dim=-1)
        tv = _tv_distance(empirical, target)
        self.assertLess(tv, 0.05, f"TV distance {tv:.4f} too large")

    def test_determinism_seeded(self):
        # Same seed -> identical draws; different seed -> different draws.
        torch.manual_seed(0)
        logits = torch.randn(256, 64)
        a = _sample(logits, 1.0, seed=123)
        b = _sample(logits, 1.0, seed=123)
        c = _sample(logits, 1.0, seed=124)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_unseeded_varies_across_calls(self):
        # seed=None uses the global RNG -> draws vary, tokens stay in range.
        torch.manual_seed(0)
        logits = torch.randn(256, 64)
        a = _sample(logits, 1.0, seed=None)
        b = _sample(logits, 1.0, seed=None)
        self.assertFalse(torch.equal(a, b))
        self.assertGreaterEqual(int(a.min()), 0)
        self.assertLess(int(a.max()), 64)

    def test_top_p_restricts_to_nucleus(self):
        # probs [0.5, 0.3, 0.15, 0.05]; top_p=0.9 keeps {0,1,2}, drops index 3.
        base = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        tokens = _sample(base.expand(5000, 4), 1.0, seed=0, top_p=0.9)
        self.assertTrue((tokens != 3).all())  # tail token never drawn
        self.assertEqual(set(tokens.tolist()), {0, 1, 2})  # nucleus covered

    def test_top_p_one_keeps_all(self):
        # top_p=1.0 -> no filtering; the tail token (index 3) is reachable.
        base = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        tokens = _sample(base.expand(20000, 4), 1.0, seed=0, top_p=1.0)
        self.assertTrue((tokens == 3).any())

    def test_top_k_restricts_to_top_k(self):
        # probs [0.5, 0.3, 0.15, 0.05]; top_k=2 keeps {0,1}.
        base = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        tokens = _sample(base.expand(5000, 4), 1.0, seed=0, top_k=2)
        self.assertTrue(torch.isin(tokens, torch.tensor([0, 1])).all())
        self.assertEqual(set(tokens.tolist()), {0, 1})

    def test_top_k_none_keeps_all(self):
        # top_k=None -> no filtering; the tail token (index 3) is reachable.
        base = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        tokens = _sample(base.expand(20000, 4), 1.0, seed=0, top_k=None)
        self.assertTrue((tokens == 3).any())

    def test_top_k_and_top_p_compose(self):
        # top_p=0.7 keeps {0,1}; top_k=1 intersects that to {0}.
        base = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        tokens = _sample(base.expand(5000, 4), 1.0, seed=0, top_p=0.7, top_k=1)
        self.assertEqual(set(tokens.tolist()), {0})


class TestSampleExport(unittest.TestCase):
    """Runtime-input semantics that survive export: temperature and seed stay
    live graph inputs (not constant-folded). Lowering/partition is covered by the
    OpTestCase classes in test_ops.py."""

    def test_runtime_temperature_single_export(self):
        # One exported program run at two temperatures (no re-export) confirms
        # temperature is a live graph input: small T is near-greedy, large T
        # spreads draws. A fixed logits row is broadcast so the spread of tokens
        # reflects the sampling entropy.
        vocab = 50
        batch = 256
        torch.manual_seed(0)
        row = torch.randn(vocab)
        logits = row.expand(batch, 1, vocab).contiguous()  # [B, S=1, vocab]
        seed = torch.tensor(0, dtype=torch.int64)

        run = torch.export.export(
            SeededSampleModel(), (logits, torch.tensor(1.0), seed), strict=True
        ).module()

        cold = run(logits, torch.tensor(1e-4), seed)
        hot = run(logits, torch.tensor(100.0), seed)

        self.assertTrue(torch.all(cold == int(torch.argmax(row))))
        self.assertEqual(cold.unique().numel(), 1)
        self.assertGreater(hot.unique().numel(), 10)

    def test_seeded_export_reproducible_no_host_rng(self):
        # Seeded export: same seed -> identical tokens across runs of one
        # exported program, independent of host RNG state (the seed is a graph
        # input, not host-side stateful RNG). Different seed -> different draws.
        torch.manual_seed(0)
        logits = torch.randn(128, 1, 64)
        seed = torch.tensor(123, dtype=torch.int64)

        run = torch.export.export(
            SeededSampleModel(), (logits, torch.tensor(1.0), seed), strict=True
        ).module()

        first = run(logits, torch.tensor(1.0), seed)
        # Perturb the host global RNG between runs; a seeded draw is unaffected.
        _ = torch.rand(1024)
        second = run(logits, torch.tensor(1.0), seed)
        self.assertTrue(torch.equal(first, second))

        other = run(logits, torch.tensor(1.0), torch.tensor(124, dtype=torch.int64))
        self.assertFalse(torch.equal(first, other))


class TestSampleEndToEnd(unittest.TestCase):
    """On-device checks whose assertions the output-compare harness can't express."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def test_top_p_end_to_end(self):
        # On-device nucleus: probs [0.5,0.3,0.15,0.05], top_p=0.9 -> token in {0,1,2}.
        logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])).view(1, 1, 4)
        inputs = (
            logits,
            torch.tensor(1.0),
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(0.9),
        )
        tmp = Path(self._tmp)
        pte, in_bin, out_bin = tmp / "topp.pte", tmp / "in.bin", tmp / "out.bin"
        export_model_to_pte(TopPSampleModel(), inputs, pte)
        save_tensors_to_bin(list(inputs), in_bin)

        self.assertTrue(run_cpp_test_runner(pte, in_bin, out_bin))
        (token,) = load_tensors_from_bin(out_bin)
        self.assertIn(int(token), {0, 1, 2})  # tail token (index 3) excluded

    def test_top_k_end_to_end(self):
        # On-device top-k: probs [0.5,0.3,0.15,0.05], top_k=2 -> token in {0,1}.
        logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])).view(1, 1, 4)
        inputs = (
            logits,
            torch.tensor(1.0),
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(2, dtype=torch.int64),
        )
        tmp = Path(self._tmp)
        pte, in_bin, out_bin = tmp / "topk.pte", tmp / "in.bin", tmp / "out.bin"
        export_model_to_pte(TopKSampleModel(), inputs, pte)
        save_tensors_to_bin(list(inputs), in_bin)

        self.assertTrue(run_cpp_test_runner(pte, in_bin, out_bin))
        (token,) = load_tensors_from_bin(out_bin)
        self.assertIn(int(token), {0, 1})  # tail tokens excluded


if __name__ == "__main__":
    unittest.main()
