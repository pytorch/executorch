# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import copy
import json
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
            calls.append(
                Call("forward", [input_ids, positions], [wrapper(input_ids, positions)])
            )
    wrapper.cache.reset()
    return Sample(calls)


class VerifySamplesTest(unittest.TestCase):
    """Writes sample sets with the Python writer and replays them through the
    C++ runner, which reads them with the C++ reader; also runs prompts
    through Hugging Face tokenizers."""

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
        write_samples(cls.path, "golden", cls.golden)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _verify(self, name: str, samples: SampleSet) -> subprocess.CompletedProcess:
        write_samples(self.path, name, samples)
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

    def _first_logits(self, tokenizer: str, prompt: str, *flags: str) -> bytes:
        path = os.path.join(tempfile.mkdtemp(dir=self._tmp.name), "logits")
        result = subprocess.run(
            [
                os.environ["RUN_HF_LLM"],
                f"--package={self.path}",
                f"--tokenizer={tokenizer}",
                f"--prompt={prompt}",
                "--max_new_tokens=1",
                f"--dump_first_logits={path}",
                *flags,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with open(path, "rb") as f:
            return f.read()

    def _modified_golden(self, edit) -> SampleSet:
        samples = copy.deepcopy(self.golden)
        edit(samples.samples[0].calls)
        return samples

    def test_replays_golden_samples(self) -> None:
        result = self._verify("replay", self.golden)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count(" ok\n"), 8)
        self.assertIn("verify:    replay, 0 calls failed", result.stdout)

    def test_replays_a_sample_that_fills_the_cache(self) -> None:
        prompt = list(range(_MAX_TOKENS))
        samples = golden_samples(self.wrapper, [prompt], _CACHE - _MAX_TOKENS)

        result = self._verify("full", samples)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count(" ok\n"), _CACHE - _MAX_TOKENS + 1)

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
        self.assertIn("verify:    wrong, 1 calls failed", result.stdout)

    def test_rejects_samples_it_cannot_replay(self) -> None:
        def out_of_vocabulary(calls: list[Call]) -> None:
            calls[1].inputs[0] = torch.tensor([[_VOCAB]])

        def wrong_position(calls: list[Call]) -> None:
            calls[1].inputs[1] = torch.tensor([99])

        def other_method(calls: list[Call]) -> None:
            calls[1].method = "decode"

        logits = torch.zeros(1, 1, _VOCAB)
        prompt = list(range(_MAX_TOKENS))
        one_call_too_many = [
            Call(
                "forward", [torch.tensor([prompt]), torch.arange(len(prompt))], [logits]
            )
        ] + [
            Call(
                "forward",
                [torch.tensor([[1]]), torch.tensor([len(prompt) + step])],
                [logits],
            )
            for step in range(_CACHE + 1 - len(prompt))
        ]

        cases = [
            (self._modified_golden(out_of_vocabulary), "outside the vocabulary"),
            (self._modified_golden(wrong_position), "not a prefill or the next decode"),
            (self._modified_golden(other_method), "call 1 runs method decode"),
            (SampleSet([Sample(one_call_too_many)]), "does not fit the method"),
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

    def test_prompts_through_hugging_face_tokenizers(self) -> None:
        # Every ID is a token: three special ones, then one character each.
        # The pre-tokenizer splits with a lookahead regex, which RE2 rejects.
        tokenizer = {
            "added_tokens": [
                {"id": i, "content": token, "special": True}
                for i, token in enumerate(["<unk>", "<s>", "</s>"])
            ],
            "pre_tokenizer": {
                "type": "Split",
                "pattern": {"Regex": r"\s+(?!\S)|\s+"},
                "behavior": "Isolated",
                "invert": False,
            },
            "model": {
                "type": "BPE",
                "vocab": {chr(i): i for i in range(3, _VOCAB)},
                "merges": [],
            },
        }
        # By default, the runner prepends the BOS tokenizer_config.json names,
        # unless it turns BOS off.
        cases = [
            ({"bos_token": "<s>", "eos_token": "</s>"}, "<s>Hi there"),
            ({"bos_token": None, "eos_token": "</s>"}, "Hi there"),
            ({"bos_token": "<s>", "add_bos_token": False}, "Hi there"),
        ]
        for config, prompt in cases:
            with self.subTest(config=config), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "tokenizer.json")
                with open(path, "w") as f:
                    json.dump(tokenizer, f)
                with open(os.path.join(tmp, "tokenizer_config.json"), "w") as f:
                    json.dump(config, f)
                self.assertEqual(
                    self._first_logits(path, "Hi there"),
                    self._first_logits(path, prompt, "--num_bos=0"),
                )
