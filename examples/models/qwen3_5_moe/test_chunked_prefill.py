# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chunked-vs-unchunked prefill equivalence test for the MLX qwen3.5 MoE .pte.

The MLX C++ runner chunks long prompts and carries the recurrent/conv state and
KV cache across chunk boundaries (qwen35_moe_engine.cpp prefill_tokens). Chunk
boundaries are easy to get subtly wrong, so this test asserts that feeding a
prompt as several sequential `forward` calls produces the same final-position
logits (and same greedy first token) as a single `forward` call.

It runs against an already-exported tiny MLX .pte (no tokenizer needed: random
token ids). Point it at the .pte via the QWEN_TINY_PTE env var, e.g.:

    python -m executorch.examples.models.qwen3_5_moe.export \
        --tiny-test --backend mlx --qlinear 4w --qlinear-group-size 32 \
        --output-dir /tmp/qwen35_moe_mlx_tiny
    QWEN_TINY_PTE=/tmp/qwen35_moe_mlx_tiny/model.pte \
        python -m pytest examples/models/qwen3_5_moe/test_chunked_prefill.py -v

The test skips (rather than fails) when the .pte env var is unset or the MLX
runtime is unavailable, so it is a no-op on non-MLX machines.
"""

import os
import unittest

import torch

PTE_ENV = "QWEN_TINY_PTE"


def _load_forward(pte_path):
    """Load a fresh program instance so mutable state starts zeroed."""
    from executorch.runtime import Runtime, Verification

    runtime = Runtime.get()
    program = runtime.load_program(pte_path, verification=Verification.Minimal)
    return program, program.load_method("forward")


def _scalar_metadata(program, name, default):
    try:
        result = program.load_method(name).execute([])
    except Exception:
        return default
    v = result[0]
    return int(v) if isinstance(v, int) else int(v.item())


def _last_logits(outputs):
    # forward returns logits shaped (1, T, vocab); take the final position.
    return outputs[0][0, -1, :]


class TestChunkedPrefill(unittest.TestCase):
    def setUp(self):
        self.pte_path = os.environ.get(PTE_ENV)
        if not self.pte_path:
            self.skipTest(f"{PTE_ENV} not set; export a tiny MLX .pte first")
        if not os.path.exists(self.pte_path):
            self.skipTest(f"{PTE_ENV}={self.pte_path} does not exist")
        try:
            import executorch.runtime  # noqa: F401
        except Exception as e:  # pragma: no cover - environment dependent
            self.skipTest(f"executorch.runtime unavailable: {e}")

    def test_chunked_prefill_matches_unchunked(self):
        # Read shapes from the model's constant methods.
        program, _ = _load_forward(self.pte_path)
        vocab_size = _scalar_metadata(program, "get_vocab_size", 256)
        max_seq_len = _scalar_metadata(program, "get_max_seq_len", 64)
        del program

        prompt_len = min(40, max_seq_len - 1)
        chunk = 8
        self.assertGreater(
            prompt_len,
            chunk,
            "prompt must exceed chunk size to exercise multiple chunks",
        )

        torch.manual_seed(0)
        tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)

        # Unchunked: one forward over the whole prompt (fresh program/state).
        _, forward_full = _load_forward(self.pte_path)
        pos_full = torch.arange(prompt_len, dtype=torch.long)
        logits_full = _last_logits(forward_full.execute([tokens, pos_full]))

        # Chunked: sequential forwards advancing input_pos, carrying state across
        # boundaries (fresh program/state).
        _, forward_chunk = _load_forward(self.pte_path)
        logits_chunk = None
        for off in range(0, prompt_len, chunk):
            end = min(off + chunk, prompt_len)
            chunk_tokens = tokens[:, off:end]
            chunk_pos = torch.arange(off, end, dtype=torch.long)
            logits_chunk = _last_logits(
                forward_chunk.execute([chunk_tokens, chunk_pos])
            )

        # Same greedy first token, and logits numerically close.
        self.assertEqual(
            int(torch.argmax(logits_full)),
            int(torch.argmax(logits_chunk)),
            "chunked prefill produced a different first token than unchunked",
        )
        torch.testing.assert_close(
            logits_chunk.to(torch.float32),
            logits_full.to(torch.float32),
            rtol=1e-2,
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
