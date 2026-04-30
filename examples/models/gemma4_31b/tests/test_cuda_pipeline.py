# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA-specific integration tests for the Gemma 4 31B-IT pipeline.

Tests pack → inference → export on a tiny model using the CUDA backend.
Backend-agnostic tests (quantize, save, load) live in ``test_pipeline.py``.

Requires CUDA.

Usage:
    python -m pytest examples/models/gemma4_31b/tests/test_cuda_pipeline.py -v
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.export import (
    export_and_lower,
    load_prequantized_model,
)
from executorch.examples.models.gemma4_31b.inference import _move_to_cuda, generate
from executorch.examples.models.gemma4_31b.model import Gemma4_31B
from executorch.examples.models.gemma4_31b.quant import (
    DEFAULT_CUDA_PACKERS,
    pack_model,
    quantize_model,
)
from executorch.examples.models.gemma4_31b.tests.test_pipeline import (
    build_hf_checkpoint,
    DEFAULT_RECIPE,
    MockTokenizer,
    save_checkpoint,
    TINY_CONFIG,
)


def _require_cuda(testcase: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        testcase.skipTest("CUDA required")


class TestCudaInference(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)

    def test_generate(self):
        """save → load → pack → generate (sampling + greedy)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir)
            model, config = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )
        _move_to_cuda(model, config)
        model.eval()
        tokenizer = MockTokenizer(TINY_CONFIG.vocab_size)

        torch.manual_seed(0)
        out = generate(model, tokenizer, prompt="hi", max_new_tokens=5, temperature=1.0)
        self.assertIsInstance(out, str)
        ids_part = out[len("<tokens:") : -1]
        ids = [int(s) for s in ids_part.split(",")]
        self.assertEqual(len(ids), 5)
        for tid in ids:
            self.assertGreaterEqual(tid, 0)
            self.assertLess(tid, TINY_CONFIG.vocab_size)

        out_greedy = generate(
            model, tokenizer, prompt="hi", max_new_tokens=3, temperature=0.0
        )
        self.assertIsInstance(out_greedy, str)
        self.assertGreater(len(out_greedy), 0)


class TestChunkedPrefill(unittest.TestCase):
    """Verify that chunked prefill matches one-token-at-a-time prefill."""

    def setUp(self):
        _require_cuda(self)

    def test_chunked_prefill_matches_sequential(self):
        """Long prompt chunked across ring buffer gives same logits as sequential."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir)
            model_seq, config = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )
            model_chunk, _ = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )

        _move_to_cuda(model_seq, config)
        _move_to_cuda(model_chunk, config)
        model_seq.eval()
        model_chunk.eval()

        buf_size = config.sliding_window * 2
        prompt_len = buf_size + 8  # exceeds buf_size
        torch.manual_seed(0)
        prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device="cuda")

        # Sequential: one token at a time (temperature=None returns logits)
        with torch.no_grad():
            for i in range(prompt_len):
                tok = prompt[:, i : i + 1]
                pos = torch.tensor([i], dtype=torch.long, device="cuda")
                logits_seq = model_seq(tok, pos, None)

        # Chunked: two chunks respecting buf_size
        with torch.no_grad():
            chunk1 = prompt[:, :buf_size]
            pos1 = torch.arange(buf_size, dtype=torch.long, device="cuda")
            model_chunk(chunk1, pos1, None)

            chunk2 = prompt[:, buf_size:]
            pos2 = torch.arange(buf_size, prompt_len, dtype=torch.long, device="cuda")
            logits_chunk = model_chunk(chunk2, pos2, None)

        # Compare last-token logits (skip sampling to avoid RNG differences).
        # Use allclose rather than equal — CUDA kernels can produce small FP
        # differences across execution shapes.
        max_diff = (logits_seq[0, -1].float() - logits_chunk[0, -1].float()).abs().max()
        self.assertTrue(
            torch.allclose(
                logits_seq[0, -1].float(),
                logits_chunk[0, -1].float(),
                atol=1e-2,
                rtol=1e-3,
            ),
            f"Chunked prefill diverged: max_diff={max_diff:.4g}",
        )


class TestCudaExport(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)

    def test_export_from_quantized_checkpoint(self):
        """--prequantized path: load → pack → export."""
        with tempfile.TemporaryDirectory() as ckpt_dir, tempfile.TemporaryDirectory() as out_dir:
            save_checkpoint(ckpt_dir)
            model, config = load_prequantized_model(
                ckpt_dir, max_seq_len=TINY_CONFIG.max_seq_len
            )
            export_and_lower(model, config, out_dir)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "model.pte")))
            ptd_files = [f for f in os.listdir(out_dir) if f.endswith(".ptd")]
            self.assertGreater(len(ptd_files), 0)

    def test_export_from_hf_checkpoint(self):
        """--model-dir path: load HF → quantize → pack → export."""
        with tempfile.TemporaryDirectory() as ckpt_dir, tempfile.TemporaryDirectory() as out_dir:
            build_hf_checkpoint(ckpt_dir)
            model, config = Gemma4_31B.from_hf_checkpoint(
                ckpt_dir, max_seq_len=TINY_CONFIG.max_seq_len
            )
            model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
            quantized, unquantized = quantize_model(model, DEFAULT_RECIPE)

            with torch.device("meta"):
                model = Gemma4_31B(config)
            pack_model(model, quantized, unquantized, DEFAULT_CUDA_PACKERS)
            model.eval()

            params = dict(model.named_parameters())
            self.assertIn("lm_head.weight", params)
            self.assertNotIn("layers.5.self_attn.v_proj.weight", params)
            export_and_lower(model, config, out_dir)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "model.pte")))


if __name__ == "__main__":
    unittest.main()
