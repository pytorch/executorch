# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic integration tests for the Gemma 4 31B-IT pipeline.

Tests the quantize → save → load roundtrip on a tiny model. No CUDA
required. Backend-specific tests (pack, inference, export) live in
``test_cuda_pipeline.py``.

Usage:
    python -m pytest examples/models/gemma4_31b/tests/test_pipeline.py -v
"""

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.model import (
    Gemma4_31B,
    Gemma4_31BConfig,
    RingKVCache,
)
from executorch.examples.models.gemma4_31b.quant import (
    load,
    QuantConfig,
    quantize_model,
    QuantRecipe,
    QuantRule,
    save,
)
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Shared fixtures — imported by test_cuda_pipeline.py.

TINY_CONFIG = Gemma4_31BConfig(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    num_global_key_value_heads=1,
    global_head_dim=128,
    attention_k_eq_v=True,
    sliding_rope_theta=10_000.0,
    full_rope_theta=1_000_000.0,
    full_partial_rotary_factor=0.25,
    rms_norm_eps=1e-6,
    hidden_activation="gelu_pytorch_tanh",
    final_logit_softcapping=30.0,
    tie_word_embeddings=True,
    sliding_window=16,
    max_seq_len=64,
)

QUANT_4W = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
QUANT_8W_PER_AXIS = QuantConfig(
    bits=8,
    group_size=TINY_CONFIG.hidden_size,
    symmetric=True,
    method="min_max",
)

DEFAULT_RECIPE = QuantRecipe(
    rules=[
        QuantRule(r"embed_tokens\.weight", QUANT_8W_PER_AXIS),
        QuantRule(r".*norm\.weight", None),
        QuantRule(r".*\.weight", QUANT_4W),
    ]
)


class MockTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, prompt: str):
        class _Encoded:
            ids = [1, 2, 3, 4]

        return _Encoded()

    def decode(self, ids):
        return "<tokens:" + ",".join(str(i) for i in ids) + ">"


def config_dict() -> dict:
    cfg = TINY_CONFIG
    return {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "num_global_key_value_heads": cfg.num_global_key_value_heads,
        "global_head_dim": cfg.global_head_dim,
        "attention_k_eq_v": cfg.attention_k_eq_v,
        "rope_parameters": {
            "sliding_attention": {"rope_theta": cfg.sliding_rope_theta},
            "full_attention": {
                "rope_theta": cfg.full_rope_theta,
                "partial_rotary_factor": cfg.full_partial_rotary_factor,
            },
        },
        "rms_norm_eps": cfg.rms_norm_eps,
        "hidden_activation": cfg.hidden_activation,
        "final_logit_softcapping": cfg.final_logit_softcapping,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "sliding_window": cfg.sliding_window,
        "layer_types": cfg.layer_types,
    }


def build_random_tiny_model() -> Gemma4_31B:
    torch.manual_seed(42)
    model = Gemma4_31B(TINY_CONFIG)
    model.to(dtype=torch.bfloat16)
    for p in model.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.02)
    model.eval()
    return model


def save_checkpoint(output_dir: str):
    model = build_random_tiny_model()
    model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
    quantized, unquantized = quantize_model(model, DEFAULT_RECIPE)
    os.makedirs(output_dir, exist_ok=True)
    save(quantized, unquantized, os.path.join(output_dir, "model.safetensors"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict(), f)


def build_hf_checkpoint(output_dir: str) -> None:
    model = build_random_tiny_model()
    sd = model.state_dict()
    sd.pop("lm_head.weight", None)
    hf_sd = {f"model.language_model.{k}": v.contiguous() for k, v in sd.items()}
    save_file(hf_sd, os.path.join(output_dir, "model.safetensors"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict(), f)


# ---------------------------------------------------------------------------
# Tests (CPU only, no backend dependency)


class TestQuantizeSaveLoadRoundtrip(unittest.TestCase):
    def test_roundtrip_preserves_weights(self):
        """quantize → save → load recovers all weights and configs."""
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        quantized, unquantized = quantize_model(model, DEFAULT_RECIPE)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.safetensors")
            save(quantized, unquantized, path)
            q_loaded, u_loaded = load(path)

        self.assertEqual(set(quantized.keys()), set(q_loaded.keys()))
        for fqn in quantized:
            self.assertEqual(quantized[fqn].config, q_loaded[fqn].config)
            self.assertTrue(torch.equal(quantized[fqn].qdata, q_loaded[fqn].qdata))
            self.assertTrue(torch.equal(quantized[fqn].scale, q_loaded[fqn].scale))

        self.assertEqual(set(unquantized.keys()), set(u_loaded.keys()))
        for fqn in unquantized:
            self.assertTrue(torch.equal(unquantized[fqn], u_loaded[fqn]))

    def test_embedding_quantized_as_int8(self):
        """embed_tokens is quantized to INT8 per-axis, not skipped."""
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        quantized, unquantized = quantize_model(model, DEFAULT_RECIPE)

        self.assertIn("embed_tokens.weight", quantized)
        self.assertNotIn("embed_tokens.weight", unquantized)
        self.assertEqual(quantized["embed_tokens.weight"].config.bits, 8)

    def test_corrupted_checkpoint_missing_key(self):
        """Renaming a key in the safetensors file makes it absent after load."""
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tmpdir)
            path = os.path.join(tmpdir, "model.safetensors")

            with safe_open(path, framework="pt", device="cpu") as f:
                header = f.metadata()
                tensors = {k: f.get_tensor(k) for k in f.keys()}
            tensors["norm.BOGUS"] = tensors.pop("norm.weight")
            save_file(tensors, path, metadata=header)

            q, u = load(path)
            self.assertNotIn("norm.weight", u)
            self.assertIn("norm.BOGUS", u)


class TestRingKVCache(unittest.TestCase):
    """Unit tests for the ring-buffer KV cache (CPU, no model needed)."""

    def _make_cache(self, window=4, heads=2, head_dim=8):
        return RingKVCache(
            max_batch_size=1, window_size=window, num_kv_heads=heads, head_dim=head_dim
        )

    def test_sequential_write_read(self):
        """Writing positions 0..buf_size-1 fills every slot exactly once."""
        cache = self._make_cache(window=4)
        buf_size = cache.buf_size  # 8
        for i in range(buf_size):
            pos = torch.tensor([i], dtype=torch.long)
            k = torch.full((1, 2, 1, 8), float(i))
            v = torch.full((1, 2, 1, 8), float(i + 100))
            k_out, v_out = cache.update(pos, k, v)
        for i in range(buf_size):
            slot = i % buf_size
            self.assertEqual(k_out[0, 0, slot, 0].item(), float(i))
            self.assertEqual(v_out[0, 0, slot, 0].item(), float(i + 100))

    def test_wraparound_overwrites_oldest(self):
        """Position buf_size overwrites slot 0 (the oldest entry)."""
        cache = self._make_cache(window=4)
        buf_size = cache.buf_size  # 8
        for i in range(buf_size + 1):
            pos = torch.tensor([i], dtype=torch.long)
            k = torch.full((1, 2, 1, 8), float(i))
            v = torch.full((1, 2, 1, 8), float(i))
            k_out, _ = cache.update(pos, k, v)
        # Slot 0 should now contain position buf_size (not 0)
        self.assertEqual(k_out[0, 0, 0, 0].item(), float(buf_size))
        # Slot 1 should still contain position 1
        self.assertEqual(k_out[0, 0, 1, 0].item(), 1.0)

    def test_multi_token_prefill(self):
        """Writing multiple positions in one call places them correctly."""
        cache = self._make_cache(window=4)
        pos = torch.arange(4, dtype=torch.long)
        k = torch.arange(4).float().view(1, 1, 4, 1).expand(1, 2, 4, 8)
        v = torch.zeros(1, 2, 4, 8)
        k_out, _ = cache.update(pos, k, v)
        for i in range(4):
            self.assertEqual(k_out[0, 0, i, 0].item(), float(i))

    def test_assert_on_oversized_prefill(self):
        """seq_len > buf_size raises AssertionError."""
        cache = self._make_cache(window=4)
        buf_size = cache.buf_size
        pos = torch.arange(buf_size + 1, dtype=torch.long)
        k = torch.zeros(1, 2, buf_size + 1, 8)
        v = torch.zeros(1, 2, buf_size + 1, 8)
        with self.assertRaises(AssertionError):
            cache.update(pos, k, v)


if __name__ == "__main__":
    unittest.main()
