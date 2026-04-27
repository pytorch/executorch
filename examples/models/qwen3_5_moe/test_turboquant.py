# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test TurboQuant KV cache on a tiny Qwen 3.5 MoE model.

Creates a tiny model (no downloads needed), quantizes weights, applies
TurboQuant KV cache compression, exports with torch.export, and verifies
the exported program produces correct output.

Requires CUDA (fused_moe and tq4_sdpa Triton kernels).

Usage:
    python -m pytest examples/models/qwen3_5_moe/test_turboquant.py -v
"""

import unittest

import torch

from executorch.examples.models.qwen3_5_moe.export import (
    _apply_turboquant,
    _materialize_buffers,
    _quantize,
)
from executorch.examples.models.qwen3_5_moe.model import Qwen35MoE, Qwen35MoEConfig
from torch.export import Dim, export


TINY_CONFIG = Qwen35MoEConfig(
    vocab_size=256,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_kv_heads=2,
    head_dim=64,
    partial_rotary_factor=0.25,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    shared_expert_intermediate_size=128,
    full_attention_interval=2,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
    max_seq_len=64,
)


def _make_model(turboquant=False):
    """Create a tiny quantized model, optionally with TurboQuant KV cache."""
    import executorch.backends.cuda.triton.kernels  # noqa: F401

    torch.manual_seed(42)
    model = Qwen35MoE(TINY_CONFIG)
    model.to(dtype=torch.bfloat16)
    for p in model.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.02)
    model.eval()

    class Args:
        qlinear = "4w"
        qembedding = None
        qlinear_group_size = 32
        hqq = False

    _quantize(model, TINY_CONFIG, Args())
    _materialize_buffers(model, TINY_CONFIG)

    if turboquant:
        _apply_turboquant(model, TINY_CONFIG)

    model.to("cuda")
    return model


def _greedy_decode(forward_fn, prompt, num_tokens):
    """Greedy decode using forward(tokens, input_pos) signature."""
    for i, tok_id in enumerate(prompt):
        logits = forward_fn(
            torch.tensor([[tok_id]], dtype=torch.long, device="cuda"),
            torch.tensor([i], dtype=torch.long, device="cuda"),
        )
    generated = []
    next_tok = logits[0, -1].argmax().item()
    generated.append(next_tok)
    for i in range(num_tokens - 1):
        logits = forward_fn(
            torch.tensor([[next_tok]], dtype=torch.long, device="cuda"),
            torch.tensor([len(prompt) + i], dtype=torch.long, device="cuda"),
        )
        next_tok = logits[0, -1].argmax().item()
        generated.append(next_tok)
    return generated


class TestTurboQuant(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_eager_quality(self):
        """TurboQuant model has >99% cosine similarity to baseline."""
        model_base = _make_model(turboquant=False)
        model_tq = _make_model(turboquant=True)

        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")
        input_pos = torch.arange(4, dtype=torch.long, device="cuda")

        with torch.no_grad():
            logits_base = model_base(tokens, input_pos)
            logits_tq = model_tq(tokens, input_pos)

        cos = torch.nn.functional.cosine_similarity(
            logits_base.reshape(1, -1).float(),
            logits_tq.reshape(1, -1).float(),
        ).item()
        self.assertGreater(cos, 0.99, f"Cosine {cos:.4f}")

    def test_eager_decode_quality(self):
        """TurboQuant decode logits stay close to baseline across steps."""
        model_base = _make_model(turboquant=False)
        model_tq = _make_model(turboquant=True)

        prompt = [1, 2, 3]
        for i, tok_id in enumerate(prompt):
            tok = torch.tensor([[tok_id]], dtype=torch.long, device="cuda")
            pos = torch.tensor([i], dtype=torch.long, device="cuda")
            with torch.no_grad():
                logits_base = model_base(tok, pos)
                logits_tq = model_tq(tok, pos)

        # Check cosine similarity of logits after prefill
        cos = torch.nn.functional.cosine_similarity(
            logits_base.reshape(1, -1).float(),
            logits_tq.reshape(1, -1).float(),
        ).item()
        self.assertGreater(cos, 0.99, f"Prefill cosine {cos:.4f}")

    def test_export_matches_eager(self):
        """Exported TQ model produces same greedy tokens as eager."""
        model = _make_model(turboquant=True)

        def eager_fn(tok, pos):
            with torch.no_grad():
                return model(tok, pos)

        eager_tokens = _greedy_decode(eager_fn, [1, 2, 3], 5)

        # Export
        seq_dim = Dim("seq_len", min=1, max=TINY_CONFIG.max_seq_len - 1)
        with torch.no_grad():
            ep = export(
                model,
                (
                    torch.tensor([[0, 1]], dtype=torch.long, device="cuda"),
                    torch.tensor([0, 1], dtype=torch.long, device="cuda"),
                ),
                dynamic_shapes=({1: seq_dim}, {0: seq_dim}),
                strict=True,
            )
        ep_mod = ep.module()

        def exported_fn(tok, pos):
            with torch.no_grad():
                return ep_mod(tok, pos)

        exported_tokens = _greedy_decode(exported_fn, [1, 2, 3], 5)
        self.assertEqual(eager_tokens, exported_tokens)

    def test_kv_cache_state_matters(self):
        """Different prefills produce different continuations."""
        model_a = _make_model(turboquant=True)
        model_b = _make_model(turboquant=True)

        def fn_a(tok, pos):
            with torch.no_grad():
                return model_a(tok, pos)

        def fn_b(tok, pos):
            with torch.no_grad():
                return model_b(tok, pos)

        tokens_a = _greedy_decode(fn_a, [1, 2, 3], 3)
        tokens_b = _greedy_decode(fn_b, [10, 20, 30], 3)
        self.assertNotEqual(tokens_a, tokens_b)

    def test_replacement_count(self):
        """_apply_turboquant replaces exactly the full-attention layers."""
        model = _make_model(turboquant=True)
        from executorch.extension.llm.modules.turboquant import TurboQuantKVCache

        tq_count = sum(
            1
            for layer in model.layers
            if isinstance(getattr(layer.attn, "kv_cache", None), TurboQuantKVCache)
        )
        fa_count = sum(1 for lt in TINY_CONFIG.layer_types if lt == "full_attention")
        self.assertEqual(tq_count, fa_count)


if __name__ == "__main__":
    unittest.main()
