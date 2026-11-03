# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The kvcache_io runner's own protocol, without a QNN export dependency."""

import unittest

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.llm.io_contracts import io_contract, KvCacheIORunner
from executorch.runtime import Runtime, Verification

try:
    from transformers import GenerationConfig, LlamaConfig
    from transformers.exporters import ExecutorchModelRunner, ExportedGenerator
except ImportError:
    ExportedGenerator = None

L = 32
PROMPTS = ([5, 17, 42, 9, 77, 3, 64], [11])


class _Step(torch.nn.Module):
    def forward(self, tokens, mask, positions, past_k, past_v):
        rows = tokens.float().view(1, 1, -1, 1)
        keys = torch.cat((past_k.transpose(2, 3), rows), 2)
        values = torch.cat((past_v, rows), 2)
        history = (keys + values).squeeze(-1).unsqueeze(2)
        next_id = (history * (mask == 0)).sum(-1).squeeze(1) % 128
        logits = -(torch.arange(128).float()[None, None] - next_id[..., None]).abs()
        return logits, rows.transpose(2, 3), rows


class TestKvCacheIORunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        steps = {}
        for query in (1, 4):
            inputs = (
                torch.zeros(1, query, dtype=torch.int32),
                torch.zeros(1, 1, query, L),
                torch.zeros(1, query, dtype=torch.int32),
                torch.zeros(1, 1, 1, L - query),
                torch.zeros(1, 1, L - query, 1),
            )
            steps[query] = torch.export.export(_Step(), inputs)
        cls.programs = {}
        for name, methods in (
            ("decode", {"kv_forward": steps[1]}),
            ("hybrid", {"kv_forward": steps[1], "prefill_forward": steps[4]}),
        ):
            cls.programs[name] = (
                to_edge_transform_and_lower(
                    methods,
                    constant_methods={
                        "get_io_contract": "kvcache_io",
                        "get_io_contract_version": 1,
                        "get_max_seq_len": L,
                        "get_n_layers": 1,
                    },
                )
                .to_executorch()
                .buffer
            )

    def _load(self, name):
        return Runtime.get().load_program(
            self.programs[name], verification=Verification.Minimal
        )

    def _expected(self, prompt, new_tokens):
        sequence = list(prompt)
        for _ in range(new_tokens):
            sequence.append(2 * sum(sequence) % 128)
        return sequence[len(prompt) :]

    def test_greedy_matches_model(self):
        for name in ("decode", "hybrid"):
            program = self._load(name)
            self.assertEqual(io_contract(program), "kvcache_io")
            runner = KvCacheIORunner([program])
            for prompt in PROMPTS:
                with self.subTest(program=name, prompt=prompt):
                    self.assertEqual(
                        runner.generate(prompt, 8), self._expected(prompt, 8)
                    )

    @unittest.skipIf(ExportedGenerator is None, "needs transformers.exporters")
    def test_exported_generator_drives_the_contract(self):
        runner = ExecutorchModelRunner._load(self.programs["hybrid"], {})
        config = LlamaConfig(vocab_size=128, eos_token_id=None)
        generator = ExportedGenerator(
            {"decoder": runner}, config, GenerationConfig(do_sample=False)
        )
        for prompt in PROMPTS:
            with self.subTest(prompt=prompt):
                out = generator.generate(
                    input_ids=torch.tensor([prompt]), max_new_tokens=8, do_sample=False
                )
                self.assertEqual(
                    out[0, len(prompt) :].tolist(), self._expected(prompt, 8)
                )


if __name__ == "__main__":
    unittest.main()
