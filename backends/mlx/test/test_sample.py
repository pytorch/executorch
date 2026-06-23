#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for mlx::sample (Gumbel-max token sampling).

Most tests exercise the op's eager reference implementation and the
export/partition/serialization path. The end-to-end test runs the exported
program through the compiled op_test_runner (see backends/mlx/test/README.md
for building it).

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
    count_mlx_delegate_segments,
    export_model_to_pte,
    get_mlx_node_counts,
)


def _ref_gumbel_max(logits: torch.Tensor, temperature: float, seed: int):
    """Independent Gumbel-max reference using the same torch RNG as the op."""
    gen = torch.Generator().manual_seed(seed)
    u = torch.rand(logits.shape, generator=gen)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits / temperature + gumbel, dim=-1)


def _tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two discrete distributions."""
    return 0.5 * torch.abs(p - q).sum().item()


def _sample(logits, temperature, seed: Optional[int]):
    t = torch.tensor(float(temperature))
    s = None if seed is None else torch.tensor(int(seed), dtype=torch.int64)
    return torch.ops.mlx.sample(logits, t, s)


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


class UnseededSampleModel(nn.Module):
    """SamplingHead with temperature as a runtime input and no seed."""

    def __init__(self):
        super().__init__()
        self.head = SamplingHead(_LogitsPassthrough())

    def forward(self, logits, temperature):
        return self.head(logits, temperature=temperature)


class TestSampleOp(unittest.TestCase):
    """Eager reference behavior of mlx::sample (no export / no runtime)."""

    def test_greedy_parity_small_temperature(self):
        # Small temperature -> Gumbel-max collapses to argmax(logits).
        torch.manual_seed(0)
        logits = torch.randn(8, 1024)
        token = _sample(logits, 1e-4, seed=0)
        self.assertTrue(torch.equal(token, torch.argmax(logits, dim=-1)))

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


class TestSampleExport(unittest.TestCase):
    """torch.export, runtime inputs, and MLXPartitioner lowering."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

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

    def test_export_strict_with_graph_inputs(self):
        # strict=True export keeps logits, temperature, and seed as graph inputs.
        logits = torch.randn(1, 4, 256)
        ep = torch.export.export(
            SeededSampleModel(),
            (logits, torch.tensor(0.8), torch.tensor(0, dtype=torch.int64)),
            strict=True,
        )
        self.assertEqual(len(ep.graph_signature.user_inputs), 3)

    def test_seeded_lowers_to_mlx_delegate(self):
        # The op is assigned to the MLX delegate, with the seed threaded in.
        pte = Path(self._tmp) / "seeded.pte"
        logits = torch.randn(1, 4, 256)
        export_model_to_pte(
            SeededSampleModel(),
            (logits, torch.tensor(0.8), torch.tensor(0, dtype=torch.int64)),
            pte,
        )
        self.assertEqual(count_mlx_delegate_segments(pte), 1)
        counts = get_mlx_node_counts(pte)
        self.assertEqual(counts.get("RandomBitsNode", 0), 1)
        self.assertEqual(counts.get("ArgmaxNode", 0), 1)
        self.assertEqual(counts.get("ItemIntNode", 0), 1)  # seed via .item()

    def test_unseeded_lowers_without_seed_field(self):
        # seed=None lowers cleanly: RandomBitsNode emitted with no seed field
        # (hence no ItemIntNode threading a seed Vid).
        pte = Path(self._tmp) / "unseeded.pte"
        logits = torch.randn(1, 4, 256)
        export_model_to_pte(UnseededSampleModel(), (logits, torch.tensor(0.8)), pte)
        self.assertEqual(count_mlx_delegate_segments(pte), 1)
        counts = get_mlx_node_counts(pte)
        self.assertEqual(counts.get("RandomBitsNode", 0), 1)
        self.assertEqual(counts.get("ItemIntNode", 0), 0)


class TestSampleEndToEnd(unittest.TestCase):
    """Run the exported program through the compiled op_test_runner."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def test_end_to_end(self):
        # Requires the compiled op_test_runner (see backends/mlx/test/README.md).
        from executorch.backends.mlx.test.test_utils import (
            load_tensors_from_bin,
            run_cpp_test_runner,
            save_tensors_to_bin,
        )

        vocab = 32
        logits = torch.randn(1, 4, vocab)
        inputs = (logits, torch.tensor(0.8), torch.tensor(0, dtype=torch.int64))

        tmp = Path(self._tmp)
        pte, in_bin, out_bin = tmp / "e2e.pte", tmp / "in.bin", tmp / "out.bin"
        export_model_to_pte(SeededSampleModel(), inputs, pte)
        save_tensors_to_bin(list(inputs), in_bin)

        self.assertTrue(run_cpp_test_runner(pte, in_bin, out_bin))
        (token,) = load_tensors_from_bin(out_bin)
        self.assertEqual(tuple(token.shape), (1,))
        self.assertTrue(0 <= int(token) < vocab)

    def test_bf16_large_vocab_greedy_parity(self):
        # Regression: bf16 logits + large vocab. A dominant logit must win under
        # near-greedy sampling. Catches the bug where casting the uniform to bf16
        # rounded the clamp (~0.99999994) up to 1.0 -> log(0) -> +inf gumbel,
        # which then beat even a huge logit and produced a constant wrong token.
        from executorch.backends.mlx.test.test_utils import (
            load_tensors_from_bin,
            run_cpp_test_runner,
            save_tensors_to_bin,
        )

        torch.manual_seed(0)
        vocab = 4000
        logits = torch.randn(1, 4, vocab, dtype=torch.bfloat16)
        logits[0, -1, 1234] = 50.0  # unambiguous argmax
        inputs = (logits, torch.tensor(1e-4), torch.tensor(0, dtype=torch.int64))

        tmp = Path(self._tmp)
        pte, in_bin, out_bin = tmp / "bf16.pte", tmp / "in.bin", tmp / "out.bin"
        export_model_to_pte(SeededSampleModel(), inputs, pte)
        save_tensors_to_bin(list(inputs), in_bin)

        self.assertTrue(run_cpp_test_runner(pte, in_bin, out_bin))
        (token,) = load_tensors_from_bin(out_bin)
        self.assertEqual(int(token), 1234)


if __name__ == "__main__":
    unittest.main()
