# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA-specific integration tests for the Muse Glimmer pipeline.

Tests pack -> inference -> export on a tiny model using the CUDA backend.
Backend-agnostic tests (quantize, save, load) live in ``test_pipeline.py``.

Requires CUDA.

Usage:
    python -m pytest examples/models/muse-glimmer/tests/test_cuda_pipeline.py -v
"""

import json
import os
import tempfile
import unittest
from dataclasses import replace

import executorch.backends.cuda.quantize_op_dispatch as _quantize_op_dispatch  # noqa: F401
import torch
from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
from executorch.examples.models.muse_glimmer.export.common import (
    mutable_buffer_metadata,
)
from executorch.examples.models.muse_glimmer.export.export_solo import (
    export_and_lower,
    load_prequantized_model,
)
from executorch.examples.models.muse_glimmer.inference import (
    _move_to_cuda,
    BOS_TOKEN_ID,
    generate,
)
from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import _finalize
from executorch.examples.models.muse_glimmer.model.model import FlatKVCache
from executorch.examples.models.muse_glimmer.source_transformations.cuda import (
    add_dflash_hidden_tapping,
    add_on_device_sampler,
    cuda_source_transformations,
)
from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
    build_random_tiny_model,
    DEFAULT_RECIPE,
    MockTokenizer,
    save_checkpoint,
    TINY_CONFIG,
)
from executorch.extension.llm.export.quant import quantize_model


def _require_cuda(testcase: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        testcase.skipTest("CUDA required")


class TestMutableBufferMetadataTest(unittest.TestCase):
    def test_combined_model_contains_target_and_draft_kv_caches(self):
        combined = torch.nn.Module()
        combined.target = torch.nn.Module()
        combined.target.kv_cache = FlatKVCache(1, 4, 2, 8)
        combined.draft = torch.nn.Module()
        combined.draft.kv_cache = FlatKVCache(1, 4, 2, 8)

        metadata = json.loads(mutable_buffer_metadata(combined))

        self.assertEqual(1, metadata["version"])
        self.assertCountEqual(
            [
                "target.kv_cache.k_cache",
                "target.kv_cache.v_cache",
                "draft.kv_cache.k_cache",
                "draft.kv_cache.v_cache",
            ],
            metadata["mutable_buffers"],
        )


class TestCudaInferenceTest(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)

    def test_generate(self):
        """save -> load -> pack -> generate produces valid tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir)
            model, config = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )
        _move_to_cuda(model)
        model.eval()
        tokenizer = MockTokenizer(TINY_CONFIG.vocab_size)

        bos = min(BOS_TOKEN_ID, TINY_CONFIG.vocab_size - 1)
        torch.manual_seed(0)
        out = generate(
            model,
            tokenizer,
            prompt="hi",
            max_new_tokens=5,
            temperature=1.0,
            bos_token_id=bos,
            eos_token_id=TINY_CONFIG.vocab_size - 1,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_generate_greedy(self):
        """Near-greedy generation (temperature=0) produces valid output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir)
            model, config = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )
        _move_to_cuda(model)
        model.eval()
        tokenizer = MockTokenizer(TINY_CONFIG.vocab_size)

        bos = min(BOS_TOKEN_ID, TINY_CONFIG.vocab_size - 1)
        out = generate(
            model,
            tokenizer,
            prompt="hi",
            max_new_tokens=3,
            temperature=0.0,
            bos_token_id=bos,
            eos_token_id=TINY_CONFIG.vocab_size - 1,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


class TestChunkedPrefillTest(unittest.TestCase):
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

        _move_to_cuda(model_seq)
        _move_to_cuda(model_chunk)
        model_seq.eval()
        model_chunk.eval()

        # Sliding window from tiny config
        window = 0
        for layer in model_seq.layers:
            if layer.is_sliding:
                window = layer.self_attn.window_size
                break
        buf_size = window * 2
        prompt_len = buf_size + 4
        torch.manual_seed(0)
        prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device="cuda")

        # Sequential: one token at a time
        with torch.no_grad():
            for i in range(prompt_len):
                tok = prompt[:, i : i + 1]
                pos = torch.tensor([i], dtype=torch.long, device="cuda")
                logits_seq = model_seq(tok, pos)

        # Chunked: two chunks
        with torch.no_grad():
            chunk1 = prompt[:, :buf_size]
            pos1 = torch.arange(buf_size, dtype=torch.long, device="cuda")
            model_chunk(chunk1, pos1)

            chunk2 = prompt[:, buf_size:]
            pos2 = torch.arange(buf_size, prompt_len, dtype=torch.long, device="cuda")
            logits_chunk = model_chunk(chunk2, pos2)

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


class TestDFlashTargetPrefillTest(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)

    def test_target_prefill_matches_target_and_continues_shared_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir)
            target_ref, config = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )
            target_prefill, _ = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )

        for model in (target_ref, target_prefill):
            _move_to_cuda(model)
            model.eval()
            cuda_source_transformations(model)
            add_dflash_hidden_tapping(model, [0])

        torch.manual_seed(7)
        prompt = torch.randint(0, config.vocab_size, (1, 7), device="cuda")
        prefill_tokens = prompt[:, :5]
        prefill_pos = torch.arange(5, dtype=torch.long, device="cuda")
        verify_tokens = prompt[:, 5:]
        verify_pos = torch.arange(5, 7, dtype=torch.long, device="cuda")

        with torch.no_grad():
            ref_logits, ref_hidden = target_ref(prefill_tokens, prefill_pos)
            prefill_logits, prefill_hidden = target_prefill.dflash_prefill_forward(
                prefill_tokens, prefill_pos
            )

            ref_verify_logits, ref_verify_hidden = target_ref(verify_tokens, verify_pos)
            shared_verify_logits, shared_verify_hidden = target_prefill(
                verify_tokens, verify_pos
            )

        self.assertEqual(prefill_logits.shape, torch.Size([1, 1, config.vocab_size]))
        torch.testing.assert_close(
            prefill_logits, ref_logits[:, -1:, :], rtol=1e-3, atol=1e-2
        )
        torch.testing.assert_close(prefill_hidden, ref_hidden, rtol=1e-3, atol=1e-2)
        torch.testing.assert_close(
            shared_verify_logits, ref_verify_logits, rtol=1e-3, atol=1e-2
        )
        torch.testing.assert_close(
            shared_verify_hidden, ref_verify_hidden, rtol=1e-3, atol=1e-2
        )


class TestCudaExportTest(unittest.TestCase):
    def setUp(self):
        _require_cuda(self)

    def test_export_from_quantized_checkpoint(self):
        """--prequantized path: load -> pack -> export."""
        from executorch.runtime import Runtime, Verification

        with (
            tempfile.TemporaryDirectory() as ckpt_dir,
            tempfile.TemporaryDirectory() as out_dir,
        ):
            save_checkpoint(ckpt_dir)
            model, config = load_prequantized_model(
                ckpt_dir, max_seq_len=TINY_CONFIG.max_seq_len
            )
            export_and_lower(model, config, out_dir)
            pte = os.path.join(out_dir, "model.pte")
            self.assertTrue(os.path.exists(pte))
            ptd_files = [f for f in os.listdir(out_dir) if f.endswith(".ptd")]
            self.assertGreater(len(ptd_files), 0)
            program = Runtime.get().load_program(pte, verification=Verification.Minimal)
            self.assertTrue(
                {
                    "embed_text",
                    "forward_from_embeddings",
                    "decode_from_embedding",
                }.issubset(program.method_names)
            )

    def test_export_from_bf16_quantize_inline(self):
        """Build bf16 model -> quantize -> pack -> export."""
        model = build_random_tiny_model()
        atomic_sd = quantize_model(model, DEFAULT_RECIPE)

        # _finalize sets config.fuse_* in place, so pass a copy of the shared
        # fixture rather than TINY_CONFIG itself.
        config = replace(TINY_CONFIG)
        model = _finalize(atomic_sd, "cuda", config, torch.bfloat16)

        with tempfile.TemporaryDirectory() as out_dir:
            export_and_lower(model, config, out_dir)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "model.pte")))


class TestInt4InferenceTest(unittest.TestCase):
    """Test Int4Tensor passthrough with dispatch override."""

    def setUp(self):
        _require_cuda(self)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir)
            self.model, self.config = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )
        _move_to_cuda(self.model)
        self.model.eval()
        # The model returns logits; enable on-device sampling for these int4
        # inference tests via the CUDA source transform.
        add_on_device_sampler(self.model)

    def _forward(self):
        with torch.no_grad():
            tok = torch.tensor([[1]], dtype=torch.long, device="cuda")
            pos = torch.tensor([0], dtype=torch.long, device="cuda")
            return self.model(tok, pos)

    def test_int4_weights_preserved(self):
        """Packing converts Int4Tensor to CudaCoalescedInt4Tensor."""
        w = self.model.layers[0].mlp.gate_up_proj.weight.data
        self.assertIsInstance(w, CudaCoalescedInt4Tensor)

    def test_inference_produces_valid_output(self):
        out = self._forward()
        self.assertEqual(out.shape, torch.Size([1, 1, self.config.vocab_size]))
        self.assertFalse(out.isnan().any())

    def test_deterministic(self):
        """Same seed produces same output."""
        torch.manual_seed(99)
        out1 = self._forward()
        # Reset KV cache by reloading
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir)
            model2, _ = load_prequantized_model(
                tmpdir, max_seq_len=TINY_CONFIG.max_seq_len
            )
        _move_to_cuda(model2)
        model2.eval()
        add_on_device_sampler(model2)
        with torch.no_grad():
            tok = torch.tensor([[1]], dtype=torch.long, device="cuda")
            pos = torch.tensor([0], dtype=torch.long, device="cuda")
            torch.manual_seed(99)
            out2 = model2(tok, pos)
        torch.testing.assert_close(out1, out2, rtol=0, atol=0)

    def test_embedding_works(self):
        tok = torch.tensor([[1]], dtype=torch.long, device="cuda")
        emb = self.model.embed_tokens(tok)
        self.assertFalse(emb.isnan().any())


if __name__ == "__main__":
    unittest.main()
