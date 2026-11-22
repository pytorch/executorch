# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
import warnings
from unittest import mock

import gguf
import numpy as np
import torch
from executorch.backends.native.examples.hf_llm import gguf_hf
from executorch.backends.native.examples.hf_llm.gguf_hf import (
    check_config_tensors,
    gguf_architecture,
    gguf_to_hf,
)
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from executorch.extension.llm.export.load import iter_checkpoint
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    GemmaConfig,
    LlamaConfig,
    MistralConfig,
    Qwen2Config,
    Qwen3_5TextConfig,
    Qwen3Config,
)
from transformers.core_model_loading import WeightConverter


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


def _tiny_muse_glimmer():
    """A 4-layer Muse Glimmer with its vision tower, or None where transformers
    has no Muse Glimmer."""
    try:
        from transformers import MuseGlimmerConfig, MuseGlimmerForConditionalGeneration
    except ImportError:
        return None
    config = MuseGlimmerConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "sliding_window": 4,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "layer_rope_theta": [10000.0] * 3 + [0],
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        },
        projector_hidden_size=32,
        out_hidden_size=32,
    )
    return MuseGlimmerForConditionalGeneration(config)


def _to_gguf_rows(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """llama.cpp's convert_hf_to_gguf.py Q/K permutation."""
    return (
        weight.reshape(heads, 2, weight.shape[0] // heads // 2, *weight.shape[1:])
        .swapaxes(1, 2)
        .reshape(weight.shape)
    )


# The GGUF type of each quantized weight of a two-layer tiny Llama by llama.cpp
# role, mixing types as "Q4_K_M" files do. A "blk.<i>.<role>" entry overrides
# the role in one layer.
_MIXED_TYPES = {
    "token_embd": "q4_k",
    "output": "q6_k",
    "attn_q": "q4_k",
    "attn_k": "q5_k",
    "attn_v": "q6_k",
    "attn_output": "q4_k",
    "ffn_gate": "q5_k",
    "ffn_up": "q4_k",
    "ffn_down": "q6_k",
    "blk.1.attn_v": "q8_0",
    "blk.1.attn_q": "q3_k",
    "blk.1.ffn_up": "q2_k",
    "blk.1.attn_output": "q4_0",
    "blk.1.ffn_gate": "q5_0",
    "blk.1.attn_k": "q5_1",
    "blk.1.ffn_down": "q4_1",
}
_LAYER_ROLES = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
)
# The HF parameter each quantized tensor of that file loads into, written out
# rather than taken from gguf_to_hf, so that a wrong mapping cannot pass on its
# own terms.
_HF_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
    "blk.0.attn_k.weight": "model.layers.0.self_attn.k_proj.weight",
    "blk.0.attn_v.weight": "model.layers.0.self_attn.v_proj.weight",
    "blk.0.attn_output.weight": "model.layers.0.self_attn.o_proj.weight",
    "blk.0.ffn_gate.weight": "model.layers.0.mlp.gate_proj.weight",
    "blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj.weight",
    "blk.0.ffn_down.weight": "model.layers.0.mlp.down_proj.weight",
    "blk.1.attn_q.weight": "model.layers.1.self_attn.q_proj.weight",
    "blk.1.attn_k.weight": "model.layers.1.self_attn.k_proj.weight",
    "blk.1.attn_v.weight": "model.layers.1.self_attn.v_proj.weight",
    "blk.1.attn_output.weight": "model.layers.1.self_attn.o_proj.weight",
    "blk.1.ffn_gate.weight": "model.layers.1.mlp.gate_proj.weight",
    "blk.1.ffn_up.weight": "model.layers.1.mlp.up_proj.weight",
    "blk.1.ffn_down.weight": "model.layers.1.mlp.down_proj.weight",
}
# The byte offsets of each type's fp16 block scales.
_SCALE_OFFSETS = {
    "q4_k": (0, 2),
    "q5_k": (0, 2),
    "q6_k": (208,),
    "q8_0": (0,),
    "q3_k": (108,),
    "q2_k": (80, 82),
    "q4_0": (0,),
    "q5_0": (0,),
    "q5_1": (0, 2),
    "q4_1": (0, 2),
}


def _random_blocks(ggml_type: str, rows: int, cols: int, seed: int) -> np.ndarray:
    """Random `ggml_type` blocks with finite fp16 scales."""
    block, block_bytes = gguf.GGML_QUANT_SIZES[
        gguf.GGMLQuantizationType[ggml_type.upper()]
    ]
    raw = np.random.default_rng(seed).integers(
        0, 256, (rows, cols // block, block_bytes), dtype=np.uint8
    )
    for offset in _SCALE_OFFSETS[ggml_type]:
        raw[..., offset : offset + 2] = np.array([2**-10], np.float16).view(np.uint8)
    return raw.reshape(rows, -1)


def _tiny_qwen35() -> torch.nn.Module:
    """Three gated DeltaNet layers, then full attention; two value heads per
    key head, so llama.cpp's tiled value heads differ from HF's grouped ones."""
    return AutoModelForCausalLM.from_config(
        Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            tie_word_embeddings=False,
        )
    )


def _tiled(grouped: torch.Tensor, head_dim: int, dim: int = 0) -> torch.Tensor:
    """llama.cpp's value-head order along `dim`: HF's head k * 2 + j, for key
    head k of 2 and its j-th value head of 2, is stored as head j * 2 + k."""
    shape = grouped.shape
    grouped = grouped.movedim(dim, 0)
    tiled = grouped.reshape(2, 2, head_dim, *grouped.shape[1:]).transpose(0, 1)
    return tiled.reshape(grouped.shape).movedim(0, dim).reshape(shape)


# llama.cpp's Qwen3.5 tensor -> (HF parameter, what its converter did to it).
# Q and K rows (2 heads of 16 each) come first in attn_qkv and ssm_conv1d.
_QWEN35_LINEAR = {
    "attn_norm.weight": ("input_layernorm.weight", lambda w: w + 1),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", lambda w: w + 1),
    "ssm_norm.weight": ("linear_attn.norm.weight", lambda w: w),
    "attn_qkv.weight": (
        "linear_attn.in_proj_qkv.weight",
        lambda w: torch.cat([w[:64], _tiled(w[64:], 16)]),
    ),
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", lambda w: _tiled(w, 16)),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", lambda w: _tiled(w, 1)),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", lambda w: _tiled(w, 1)),
    "ssm_conv1d.weight": (
        "linear_attn.conv1d.weight",
        lambda w: torch.cat([w[:64], _tiled(w[64:], 16)]).squeeze(1),
    ),
    "ssm_a": ("linear_attn.A_log", lambda w: _tiled(-w.exp(), 1)),
    "ssm_dt.bias": ("linear_attn.dt_bias", lambda w: _tiled(w, 1)),
    "ssm_out.weight": ("linear_attn.out_proj.weight", lambda w: _tiled(w, 16, 1)),
}
_QWEN35_FULL = {
    "attn_norm.weight": ("input_layernorm.weight", lambda w: w + 1),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", lambda w: w + 1),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", lambda w: w + 1),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", lambda w: w + 1),
    "attn_q.weight": ("self_attn.q_proj.weight", lambda w: w),
    "attn_k.weight": ("self_attn.k_proj.weight", lambda w: w),
    "attn_v.weight": ("self_attn.v_proj.weight", lambda w: w),
    "attn_output.weight": ("self_attn.o_proj.weight", lambda w: w),
}
_QWEN35_MLP = {
    f"ffn_{p}.weight": (f"mlp.{p}_proj.weight", lambda w: w)
    for p in ("gate", "up", "down")
}
_QWEN35_GLOBAL = {
    "token_embd.weight": ("model.embed_tokens.weight", lambda w: w),
    "output.weight": ("lm_head.weight", lambda w: w),
    "output_norm.weight": ("model.norm.weight", lambda w: w + 1),
}


def _write_qwen35_gguf(
    model: torch.nn.Module, path: str
) -> tuple[dict[str, str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Write `model` as llama.cpp lays a Qwen3.5 GGUF out, with Q8_0 blocks on
    layer 0's `in_proj_qkv`, `in_proj_z`, `out_proj` and `ffn_up`, and a
    multi-token prediction block. Returns the GGUF -> HF names, every stored
    tensor, and the HF-order blocks of the quantized ones."""
    params = model.state_dict()
    names: dict[str, str] = {}
    tables: dict[str, dict] = {}
    for i in range(4):
        table = (_QWEN35_LINEAR if i < 3 else _QWEN35_FULL) | _QWEN35_MLP
        for role, (param, _) in table.items():
            names[f"blk.{i}.{role}"] = f"model.layers.{i}.{param}"
            tables[f"blk.{i}.{role}"] = table
    for name, (param, _) in _QWEN35_GLOBAL.items():
        names[name] = param
        tables[name] = _QWEN35_GLOBAL
    stored = {}
    for name, param in names.items():
        role = name.split(".", 2)[-1] if name.startswith("blk.") else name
        stored[name] = tables[name][role][1](params[param]).float().numpy()
    roles = ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ffn_up.weight")
    blocks = {
        f"blk.0.{role}": _random_blocks(
            "q8_0", *params[names[f"blk.0.{role}"]].shape, seed
        )
        for seed, role in enumerate(roles)
    }
    for name, hf_raw in blocks.items():
        rows = torch.from_numpy(hf_raw)
        if name == "blk.0.attn_qkv.weight":
            rows = torch.cat([rows[:64], _tiled(rows[64:], 16)])
        elif name == "blk.0.attn_gate.weight":
            rows = _tiled(rows, 16)
        stored[name] = rows.numpy()
    stored["blk.4.nextn.eh_proj.weight"] = np.zeros((64, 128), np.float32)
    stored["blk.4.attn_norm.weight"] = np.ones(64, np.float32)
    writer = gguf.GGUFWriter(path, "qwen35")
    for name, value in stored.items():
        q8_0 = gguf.GGMLQuantizationType.Q8_0 if name in blocks else None
        writer.add_tensor(name, value, raw_dtype=q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return names, stored, blocks


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

    def test_truncated_model_skips_later_blocks(self) -> None:
        key_map, _ = gguf_to_hf(_tiny(LlamaConfig), "llama", truncated=True)

        self.assertIsNone(key_map("blk.2.attn_q.weight"))
        self.assertEqual(
            key_map("blk.1.attn_q.weight"), "model.layers.1.self_attn.q_proj.weight"
        )
        with self.assertRaisesRegex(ValueError, "maps to no llama parameter"):
            key_map("blk.0.attn_qkv.weight")

    def test_skips_multimodal_encoder_tensors(self) -> None:
        key_map, _ = gguf_to_hf(_tiny(LlamaConfig), "llama")
        encoder = (
            "v.blk.0.attn_q.weight",
            "mm.input_projection.weight",
            "a.conv1d.0.weight",
        )

        # Not assertWarns: it reads every module in sys.modules, and transformers
        # 5's lazy modules import optional dependencies on attribute access.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual([key_map(n) for n in encoder], [None] * len(encoder))
        self.assertEqual(sum("text decoder only" in str(w.message) for w in caught), 1)
        self.assertEqual(key_map("output_norm.weight"), "model.norm.weight")
        with self.assertRaisesRegex(ValueError, "maps to no llama parameter"):
            key_map("vblk.0.attn_q.weight")

    def test_checks_the_architecture_against_the_model(self) -> None:
        gguf_to_hf(_tiny(MistralConfig), "llama")
        with self.assertRaisesRegex(ValueError, "cannot load a 'qwen2' model"):
            gguf_to_hf(_tiny(Qwen2Config), "llama")
        with self.assertRaisesRegex(ValueError, "unsupported GGUF architecture"):
            gguf_to_hf(_tiny(LlamaConfig), "phi3")
        with self.assertRaisesRegex(ValueError, "cannot load a 'llama' model"):
            gguf_to_hf(_tiny(LlamaConfig), "qwen35")

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

    def test_maps_muse_glimmer_names(self) -> None:
        model = _tiny_muse_glimmer()
        if model is None:
            self.skipTest("transformers has no Muse Glimmer")
        key_map, _ = gguf_to_hf(model, "muse-glimmer")
        text = "model.language_model"
        names = {
            "token_embd.weight": f"{text}.embed_tokens.weight",
            "output_norm.weight": f"{text}.norm.weight",
            "output.weight": "lm_head.weight",
            "blk.3.attn_gate.weight": f"{text}.layers.3.self_attn.gate_proj.weight",
            "blk.0.ffn_norm.weight": f"{text}.layers.0.pre_feedforward_layernorm.weight",
            "blk.1.post_ffw_norm.weight": (
                f"{text}.layers.1.post_feedforward_layernorm.weight"
            ),
            "blk.2.attn_q_norm.weight": None,
            "blk.2.attn_k_norm.weight": None,
        }

        self.assertEqual({n: key_map(n) for n in names}, names)
        params = dict(model.named_parameters())
        self.assertTrue(all(fqn in params for fqn in names.values() if fqn))
        with self.assertRaisesRegex(ValueError, "maps to no muse_glimmer_text"):
            key_map("blk.4.attn_q.weight")

    def test_restores_muse_glimmer_weights(self) -> None:
        model = _tiny_muse_glimmer()
        if model is None:
            self.skipTest("transformers has no Muse Glimmer")
        _, convert = gguf_to_hf(model, "muse-glimmer")
        text = "model.language_model"
        q = torch.randn(64, 64)
        decoder_norm = torch.randn(64)
        final_norm = torch.randn(64)

        self.assertTrue(
            torch.equal(
                convert(
                    f"{text}.layers.0.self_attn.q_proj.weight", _to_gguf_rows(q, 4)
                ),
                q,
            )
        )
        torch.testing.assert_close(
            convert(f"{text}.layers.0.input_layernorm.weight", decoder_norm + 1),
            decoder_norm,
        )
        self.assertIs(convert(f"{text}.norm.weight", final_norm), final_norm)

    def test_checks_muse_glimmer_qk_norms_against_the_config(self) -> None:
        model = _tiny_muse_glimmer()
        if model is None:
            self.skipTest("transformers has no Muse Glimmer")
        scale = torch.full((16,), model.config.get_text_config().qk_scale_factor)
        ones = torch.ones(16)
        stored = [
            ("blk.0.attn_q_norm.weight", scale),
            ("blk.0.attn_k_norm.weight", ones),
            ("blk.0.attn_q.weight", torch.zeros(64, 64)),
        ]

        checked = list(check_config_tensors(model, "muse-glimmer", stored))

        self.assertEqual([n for n, _ in checked], [n for n, _ in stored])
        for name, value in (
            ("blk.1.attn_q_norm.weight", ones),
            ("blk.1.attn_k_norm.weight", scale),
        ):
            with self.assertRaisesRegex(ValueError, name):
                list(check_config_tensors(model, "muse-glimmer", [(name, value)]))

    def test_keeps_qwen_weights_as_stored(self) -> None:
        _, convert = gguf_to_hf(_tiny(Qwen2Config), "qwen2")
        weight = torch.randn(64, 64)

        self.assertIs(convert("model.layers.0.self_attn.q_proj.weight", weight), weight)

    def test_loads_qwen35_through_transformers_tables(self) -> None:
        torch.manual_seed(0)
        model = _tiny_qwen35()
        params = model.state_dict()
        key_map, convert = gguf_to_hf(model, "qwen35")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.gguf")
            names, stored, blocks = _write_qwen35_gguf(model, path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = dict(iter_checkpoint(path, key_map=key_map, convert=convert))

        self.assertEqual(set(loaded), set(params))
        self.assertEqual(sum("multi-token" in str(w.message) for w in caught), 1)
        for name, param in names.items():
            with self.subTest(name):
                value = loaded[param]
                if name in blocks and name != "blk.0.ssm_out.weight":
                    self.assertIsInstance(value, ExportableGGUFTensor)
                    self.assertTrue(
                        torch.equal(value.raw, torch.from_numpy(blocks[name]))
                    )
                elif name == "blk.0.ssm_out.weight":
                    tiled = gguf.quants.dequantize(
                        stored[name], gguf.GGMLQuantizationType.Q8_0
                    )
                    want = (
                        torch.from_numpy(tiled)
                        .reshape(64, 2, 2, 16)
                        .transpose(1, 2)
                        .reshape(64, 64)
                    )
                    self.assertNotIsInstance(value, ExportableGGUFTensor)
                    self.assertTrue(torch.equal(value, want))
                else:
                    torch.testing.assert_close(value, params[param].float())

    def test_truncated_qwen35_skips_later_blocks(self) -> None:
        model = _tiny_qwen35()
        key_map, _ = gguf_to_hf(model, "qwen35", truncated=True)

        # A 10-layer file's blocks past the model's 4 layers, its multi-token
        # prediction block among them.
        for name in ("blk.4.attn_norm.weight", "blk.5.ffn_up.weight"):
            self.assertIsNone(key_map(name))
        self.assertEqual(
            key_map("blk.3.ffn_up.weight"), "model.layers.3.mlp.up_proj.weight"
        )
        key_map, _ = gguf_to_hf(model, "qwen35")
        with self.assertRaisesRegex(ValueError, "maps to no qwen3_5_text parameter"):
            key_map("blk.5.ffn_up.weight")

    def test_refuses_a_transformers_mapping_with_unknown_ops(self) -> None:
        class Scale:
            supports_packed = True

        mapping = [WeightConverter("ssm_out", "out_proj", operations=[Scale()])]
        with mock.patch.object(
            gguf_hf, "get_gguf_conversion_mapping", return_value=mapping
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Scale, which export does not know"
            ):
                gguf_to_hf(_tiny_qwen35(), "qwen35")

    def test_loads_a_mixed_type_gguf(self) -> None:
        model = AutoModelForCausalLM.from_config(
            LlamaConfig(
                vocab_size=32,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
            )
        )
        key_map, convert = gguf_to_hf(model, "llama")
        params = model.state_dict()
        names = ["token_embd", "output"]
        names += [f"blk.{i}.{role}" for i in range(2) for role in _LAYER_ROLES]
        types = {
            f"{name}.weight": _MIXED_TYPES.get(name, _MIXED_TYPES[name.split(".")[-1]])
            for name in names
        }
        self.assertEqual(set(types), set(_HF_NAMES))
        # Distinct bytes for every tensor, so that a weight loaded under another
        # tensor's name fails the byte comparison below.
        hf_blocks = {
            name: _random_blocks(ggml_type, *params[_HF_NAMES[name]].shape, seed)
            for seed, (name, ggml_type) in enumerate(types.items())
        }
        heads = {f"blk.{i}.attn_q.weight": 4 for i in range(2)}
        heads |= {f"blk.{i}.attn_k.weight": 2 for i in range(2)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.gguf")
            writer = gguf.GGUFWriter(path, "llama")
            norms = ["output_norm"]
            norms += [
                f"blk.{i}.{n}" for i in range(2) for n in ("attn_norm", "ffn_norm")
            ]
            for norm in norms:
                writer.add_tensor(f"{norm}.weight", np.ones(256, np.float32))
            for name, ggml_type in types.items():
                rows = torch.from_numpy(hf_blocks[name])
                if name in heads:
                    rows = _to_gguf_rows(rows, heads[name])
                writer.add_tensor(
                    name,
                    rows.numpy(),
                    raw_dtype=gguf.GGMLQuantizationType[ggml_type.upper()],
                )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            loaded = dict(iter_checkpoint(path, key_map=key_map, convert=convert))

        self.assertEqual(set(loaded), set(params))
        for name, ggml_type in types.items():
            with self.subTest(name):
                self.assertEqual(key_map(name), _HF_NAMES[name])
                value = loaded[_HF_NAMES[name]]
                self.assertIsInstance(value, ExportableGGUFTensor)
                self.assertEqual(value.ggml_type, ggml_type)
                self.assertEqual(value.shape, params[_HF_NAMES[name]].shape)
                self.assertTrue(
                    torch.equal(value.raw, torch.from_numpy(hf_blocks[name]))
                )
