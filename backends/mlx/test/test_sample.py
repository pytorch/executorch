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


class TestSampleExport(unittest.TestCase):
    """torch.export and runtime-input semantics of the sampling head."""

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


class TestSampleEndToEnd(unittest.TestCase):
    """On-device checks whose assertions the output-compare harness can't express."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def test_bf16_large_vocab_greedy_parity(self):
        # Regression: bf16 logits + large vocab. A dominant logit must win under
        # near-greedy sampling. Catches the bug where casting the uniform to bf16
        # rounded the clamp (~0.99999994) up to 1.0 -> log(0) -> +inf gumbel,
        # which then beat even a huge logit and produced a constant wrong token.
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

    def test_greedy_temperature_zero_end_to_end(self):
        # temperature=0 takes the IfNode greedy branch -> exact argmax on device.
        torch.manual_seed(0)
        vocab = 64
        logits = torch.randn(1, 4, vocab)
        inputs = (logits, torch.tensor(0.0), torch.tensor(0, dtype=torch.int64))

        tmp = Path(self._tmp)
        pte, in_bin, out_bin = tmp / "greedy.pte", tmp / "in.bin", tmp / "out.bin"
        export_model_to_pte(SeededSampleModel(), inputs, pte)
        save_tensors_to_bin(list(inputs), in_bin)

        self.assertTrue(run_cpp_test_runner(pte, in_bin, out_bin))
        (token,) = load_tensors_from_bin(out_bin)
        self.assertEqual(int(token), int(torch.argmax(logits[0, -1])))

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


if __name__ == "__main__":
    unittest.main()
