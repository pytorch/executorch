# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import copy
import os
import subprocess
import tempfile
import unittest

import torch
from executorch.backends.native.examples.hf_llm.export_hf_llm import (
    golden_samples,
    LastTokenLM,
)
from executorch.exir.native import Call, Sample, SampleSet, to_native, write_samples
from transformers import AutoModelForCausalLM, LlamaConfig

_VOCAB = 128
_CACHE = 40
_MAX_TOKENS = 31


def _tiny_lm() -> LastTokenLM:
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
    model.config._attn_implementation = "sdpa"
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    return LastTokenLM(model, _CACHE).eval()


def _forced(wrapper: LastTokenLM, prompt: list[int], forced: list[int]) -> Sample:
    """A prefill of `prompt`, then one decode call per `forced` token."""
    wrapper.cache.reset()
    input_ids = torch.tensor([prompt])
    positions = torch.arange(len(prompt))
    calls = []
    with torch.no_grad():
        for token in [None, *forced]:
            if token is not None:
                input_ids = torch.tensor([[token]])
                positions = positions[-1:] + 1
            calls.append(Call([input_ids, positions], [wrapper(input_ids, positions)]))
    wrapper.cache.reset()
    return Sample(calls)


class VerifySamplesTest(unittest.TestCase):
    """Writes sample sets with the Python writer and replays them through the
    C++ runner, which reads them with the C++ reader."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        # Byte-level TikToken: one rank per byte, so token IDs below 256 decode.
        cls.tokenizer = os.path.join(cls._tmp.name, "tokenizer.model")
        with open(cls.tokenizer, "w") as f:
            for i in range(256):
                f.write(f"{base64.b64encode(bytes([i])).decode()} {i}\n")

        torch.manual_seed(0)
        cls.wrapper = _tiny_lm()
        seq = torch.export.Dim("seq", min=1, max=_MAX_TOKENS)
        with torch.no_grad():
            ep = torch.export.export(
                cls.wrapper,
                (torch.zeros((1, 6), dtype=torch.long), torch.arange(6)),
                dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                strict=True,
            )
        cls.path = os.path.join(cls._tmp.name, "m.ptn")
        to_native(ep).save(cls.path)
        cls.golden = golden_samples(cls.wrapper, [[3, 4, 5, 6, 7, 8], [9, 10]], 3)
        write_samples(cls.path, "forward", "golden", cls.golden)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _verify(self, name: str, samples: SampleSet) -> subprocess.CompletedProcess:
        write_samples(self.path, "forward", name, samples)
        return subprocess.run(
            [
                os.environ["RUN_HF_LLM"],
                f"--package={self.path}",
                f"--tokenizer={self.tokenizer}",
                f"--verify_samples={name}",
            ],
            capture_output=True,
            text=True,
        )

    def _modified_golden(self, edit) -> SampleSet:
        samples = copy.deepcopy(self.golden)
        edit(samples.samples[0].calls)
        return samples

    def test_replays_golden_samples(self) -> None:
        result = self._verify("replay", self.golden)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count(" ok\n"), 8)
        self.assertIn("forward/replay, 0 calls failed", result.stdout)

    def test_forces_recorded_tokens_over_its_own_choice(self) -> None:
        forced = [17, 42, 99]
        sample = _forced(self.wrapper, [3, 4, 5], forced)
        argmax = [c.outputs[0][0, -1].argmax().item() for c in sample.calls[:-1]]
        self.assertNotEqual(argmax, forced)

        result = self._verify("forced", SampleSet([sample], tol=(1e-3, 1e-3)))

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count(" ok\n"), 4)

    def test_fails_a_call_whose_logits_differ(self) -> None:
        def perturb(calls: list[Call]) -> None:
            calls[1].outputs = [calls[1].outputs[0] + 1.0]

        result = self._verify("wrong", self._modified_golden(perturb))

        self.assertEqual(result.returncode, 1)
        self.assertRegex(result.stdout, r"sample 0 call 1: max_abs_error=1 .* FAIL")
        self.assertIn("forward/wrong, 1 calls failed", result.stdout)

    def test_rejects_samples_it_cannot_replay(self) -> None:
        def out_of_vocabulary(calls: list[Call]) -> None:
            calls[1].inputs[0] = torch.tensor([[_VOCAB]])

        def wrong_position(calls: list[Call]) -> None:
            calls[1].inputs[1] = torch.tensor([99])

        cases = [
            (self._modified_golden(out_of_vocabulary), "outside the vocabulary"),
            (self._modified_golden(wrong_position), "not a prefill or the next decode"),
            (
                golden_samples(self.wrapper, [list(range(_MAX_TOKENS + 1))], 0),
                "prefill exceeds",
            ),
        ]
        for i, (samples, message) in enumerate(cases):
            with self.subTest(message=message):
                result = self._verify(f"invalid{i}", samples)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
