# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from executorch.examples.models.llama.runner.generation import LlamaRunner


class _Tokenizer:
    n_words = 100
    eos_id = 99

    def decode_token(self, token: int) -> str:
        return str(token)


class _RecordingRunner(LlamaRunner):
    def __init__(
        self,
        *,
        use_kv_cache: bool,
        enable_dynamic_shape: bool,
    ):
        with patch(
            "executorch.examples.models.llama.runner.generation.get_tokenizer",
            return_value=_Tokenizer(),
        ):
            super().__init__(
                tokenizer_path="unused",
                max_seq_len=5,
                max_batch_size=1,
                use_kv_cache=use_kv_cache,
                vocab_size=100,
                enable_dynamic_shape=enable_dynamic_shape,
            )
        self.calls = []

    def forward(self, tokens, input_pos=None):
        if (
            self.use_kv_cache
            and not self.enable_dynamic_shape
            and tokens.shape != (1, 1)
        ):
            raise RuntimeError(
                f"static input requires shape (1, 1), got {tokens.shape}"
            )
        self.calls.append(
            (
                tokens.flatten().tolist(),
                None if input_pos is None else input_pos.item(),
            )
        )
        last_token = tokens[0, -1].item()
        sampled_token = 20 if last_token == 12 else 99 if last_token == 20 else 0
        logits = torch.full((1, 100), -1.0)
        logits[0, sampled_token] = 1.0
        return logits


class GenerationTest(unittest.TestCase):
    def test_static_kv_cache_prefills_one_token_at_a_time(self):
        runner = _RecordingRunner(use_kv_cache=True, enable_dynamic_shape=False)

        generated = runner.generate(
            prompt_tokens=[10, 11, 12],
            max_seq_len=5,
            temperature=0,
            pos_base=7,
        )

        self.assertEqual(generated, [20, 99])
        self.assertEqual(
            runner.calls,
            [([10], 7), ([11], 8), ([12], 9), ([20], 10)],
        )

    def test_dynamic_kv_cache_preserves_parallel_prefill(self):
        runner = _RecordingRunner(use_kv_cache=True, enable_dynamic_shape=True)

        generated = runner.generate(
            prompt_tokens=[10, 11, 12],
            max_seq_len=5,
            temperature=0,
            pos_base=7,
        )

        self.assertEqual(generated, [20, 99])
        self.assertEqual(runner.calls, [([10, 11, 12], 7), ([20], 10)])

    def test_static_non_kv_cache_preserves_full_sequence_calls(self):
        runner = _RecordingRunner(use_kv_cache=False, enable_dynamic_shape=False)

        generated = runner.generate(
            prompt_tokens=[10, 11, 12],
            max_seq_len=5,
            temperature=0,
            pos_base=7,
        )

        self.assertEqual(generated, [20, 99])
        self.assertEqual(
            runner.calls,
            [([10, 11, 12], None), ([10, 11, 12, 20], None)],
        )


if __name__ == "__main__":
    unittest.main()
