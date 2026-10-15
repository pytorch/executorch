# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import gguf
import numpy as np
import torch
from executorch.backends.native.examples.hf.gguf_hf import gguf_architecture, gguf_to_hf
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    GemmaConfig,
    LlamaConfig,
    MistralConfig,
    Qwen2Config,
    Qwen3Config,
)


def _tiny(config_cls, heads: int = 4, kv_heads: int = 2, head_dim: int = 16, **kw):
    config = config_cls(
        vocab_size=32,
        hidden_size=heads * head_dim,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        **kw,
    )
    return AutoModelForCausalLM.from_config(config)


def _to_gguf_rows(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """llama.cpp's convert_hf_to_gguf.py Q/K permutation."""
    return (
        weight.reshape(heads, 2, weight.shape[0] // heads // 2, *weight.shape[1:])
        .swapaxes(1, 2)
        .reshape(weight.shape)
    )


class GgufHfTest(unittest.TestCase):
    def test_maps_standard_gguf_names(self) -> None:
        cases = {
            ("llama", LlamaConfig): {
                "token_embd.weight": "model.embed_tokens.weight",
                "output_norm.weight": "model.norm.weight",
                "output.weight": "lm_head.weight",
                "blk.1.attn_q.weight": "model.layers.1.self_attn.q_proj.weight",
                "blk.0.ffn_norm.weight": "model.layers.0.post_attention_layernorm.weight",
            },
            ("qwen2", Qwen2Config): {
                "blk.0.attn_q.bias": "model.layers.0.self_attn.q_proj.bias",
            },
            ("qwen3", Qwen3Config): {
                "blk.1.attn_k_norm.weight": "model.layers.1.self_attn.k_norm.weight",
            },
            ("gemma2", Gemma2Config): {
                "blk.0.post_attention_norm.weight": "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight": "model.layers.0.pre_feedforward_layernorm.weight",
                "blk.0.post_ffw_norm.weight": "model.layers.0.post_feedforward_layernorm.weight",
            },
        }
        for (arch, config_cls), names in cases.items():
            with self.subTest(arch):
                key_map, _ = gguf_to_hf(_tiny(config_cls), arch)
                self.assertEqual({n: key_map(n) for n in names}, names)

    def test_skips_derived_and_rejects_unknown_tensors(self) -> None:
        key_map, _ = gguf_to_hf(_tiny(LlamaConfig), "llama")

        self.assertIsNone(key_map("rope_freqs.weight"))
        for name in ("blk.0.attn_qkv.weight", "blk.2.attn_q.weight"):
            with self.assertRaisesRegex(ValueError, "maps to no llama parameter"):
                key_map(name)

    def test_checks_the_architecture_against_the_model(self) -> None:
        gguf_to_hf(_tiny(MistralConfig), "llama")
        with self.assertRaisesRegex(ValueError, "cannot load a 'qwen2' model"):
            gguf_to_hf(_tiny(Qwen2Config), "llama")
        with self.assertRaisesRegex(ValueError, "unsupported GGUF architecture"):
            gguf_to_hf(_tiny(LlamaConfig), "phi3")

    def test_reads_the_architecture_of_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.gguf")
            writer = gguf.GGUFWriter(path, "qwen2")
            writer.add_tensor("token_embd.weight", np.zeros((2, 4), np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            self.assertEqual(gguf_architecture(path), "qwen2")

    def test_restores_rows_from_literal_gguf_order(self) -> None:
        # convert_hf_to_gguf.py interleaves each head's two rotary halves: HF
        # rows [0, 1, 2, 3, 4, 5] of a 6-row head are stored as [0, 3, 1, 4, 2, 5].
        _, convert = gguf_to_hf(_tiny(LlamaConfig, 2, 2, 6), "llama")
        gguf_rows = torch.tensor([0.0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11])[:, None]

        restored = convert("model.layers.0.self_attn.q_proj.weight", gguf_rows)

        self.assertEqual(restored.flatten().tolist(), list(range(12)))

    def test_restores_hf_rows_of_grouped_projections_and_biases(self) -> None:
        heads, kv_heads, head_dim = 4, 2, 6
        _, convert = gguf_to_hf(
            _tiny(LlamaConfig, heads, kv_heads, head_dim, attention_bias=True),
            "llama",
        )
        cases = {
            "model.layers.0.self_attn.q_proj.weight": (heads, (heads * head_dim, 3)),
            "model.layers.0.self_attn.k_proj.weight": (
                kv_heads,
                (kv_heads * head_dim, 3),
            ),
            "model.layers.1.self_attn.k_proj.bias": (kv_heads, (kv_heads * head_dim,)),
        }
        for fqn, (n, shape) in cases.items():
            with self.subTest(fqn):
                hf = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
                self.assertTrue(torch.equal(convert(fqn, _to_gguf_rows(hf, n)), hf))
        with self.assertRaisesRegex(
            ValueError, r"expected 24 rows, got shape \(18, 3\)"
        ):
            convert("model.layers.0.self_attn.q_proj.weight", torch.zeros(18, 3))

    def test_rejects_an_odd_head_dim(self) -> None:
        # Newer transformers reject an odd head_dim when building the config.
        model = _tiny(LlamaConfig, 2, 2, 6)
        model.config.head_dim = 5

        with self.assertRaisesRegex(ValueError, "even head_dim, got 5"):
            gguf_to_hf(model, "llama")

    def test_restores_gguf_blocks_byte_exactly(self) -> None:
        kv_heads, head_dim, block_bytes = 2, 4, 144
        _, convert = gguf_to_hf(_tiny(LlamaConfig, 8, kv_heads, head_dim), "llama")
        rows = kv_heads * head_dim
        hf_raw = (
            torch.arange(rows, dtype=torch.uint8)[:, None].repeat(1, block_bytes) * 3
        )
        stored = ExportableGGUFTensor.from_raw(
            _to_gguf_rows(hf_raw, kv_heads), "q4_k", torch.float32
        )

        restored = convert("model.layers.1.self_attn.k_proj.weight", stored)

        self.assertIsInstance(restored, ExportableGGUFTensor)
        self.assertEqual(restored.ggml_type, "q4_k")
        self.assertEqual(restored.orig_dtype, torch.float32)
        self.assertTrue(torch.equal(restored.raw, hf_raw))

    def test_removes_the_gemma_norm_offset(self) -> None:
        _, convert = gguf_to_hf(_tiny(GemmaConfig), "gemma")
        weight = torch.randn(8, 4)

        norm = convert(
            "model.layers.0.input_layernorm.weight",
            torch.tensor([1.0, 1.5, 0.25], dtype=torch.float16),
        )

        self.assertEqual(norm.dtype, torch.float32)
        self.assertEqual(norm.tolist(), [0.0, 0.5, -0.75])
        self.assertIs(convert("model.layers.0.self_attn.q_proj.weight", weight), weight)
        with self.assertRaisesRegex(ValueError, "unquantized norm"):
            convert(
                "model.norm.weight",
                ExportableGGUFTensor.from_raw(
                    torch.zeros(1, 144, dtype=torch.uint8), "q4_k", torch.float32
                ),
            )

    def test_keeps_qwen_weights_as_stored(self) -> None:
        _, convert = gguf_to_hf(_tiny(Qwen2Config), "qwen2")
        weight = torch.randn(64, 64)

        self.assertIs(convert("model.layers.0.self_attn.q_proj.weight", weight), weight)
