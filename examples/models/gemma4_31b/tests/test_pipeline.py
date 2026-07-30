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
from executorch.extension.llm.export.quant import (
    QuantConfig,
    quantize_model,
    quantize_stream,
    QuantRecipe,
    QuantRule,
    to_default,
)
from safetensors import safe_open
from safetensors.torch import save_file
from torchao.prototype.safetensors.safetensors_support import (
    flatten_tensor_state_dict,
    unflatten_tensor_state_dict,
)


# ---------------------------------------------------------------------------
# Shared fixtures — imported by test_cuda_pipeline.py.

TINY_CONFIG = Gemma4_31BConfig(
    vocab_size=256,
    # hidden_size must be a multiple of 256: CUDA int4/int6 superscale packing
    # uses a per-256-weight scale step, so in-features (K) must be divisible by
    # 256. Keeps the CUDA pipeline tests on the same shared fixture.
    hidden_size=256,
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
    # On-disk producer: quantize_stream yields the torchao-native serialization
    # form (Int4Tensor) that flatten_tensor_state_dict can write. quantize_model
    # would yield the in-memory ExportableInt4Tensor, which torchao can't flatten.
    params = ((fqn, p.data) for fqn, p in model.named_parameters())
    state_dict = dict(quantize_stream(params, DEFAULT_RECIPE))
    persistent_keys = set(model.state_dict().keys())
    for fqn, buf in model.named_buffers():
        if fqn in persistent_keys and fqn not in state_dict:
            state_dict[fqn] = buf.data
    os.makedirs(output_dir, exist_ok=True)
    td, md = flatten_tensor_state_dict(state_dict)
    save_file(td, os.path.join(output_dir, "model.safetensors"), metadata=md)
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


# GGUF-friendly tiny config: Q4_K/Q6_K need in-features that are multiples of 256,
# so hidden/intermediate are 256. Two layers exercises a sliding + a global layer.
GGUF_CONFIG = Gemma4_31BConfig(
    vocab_size=256,
    hidden_size=256,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=256,
    global_head_dim=512,
    sliding_window=16,
    max_seq_len=64,
)


def _model_to_gguf_key(fqn: str):
    """Invert ``gguf_loader._KEY_MAP`` (model FQN -> GGUF tensor name)."""
    from executorch.examples.models.gemma4_31b.gguf_loader import _KEY_MAP

    for gguf_pat, model_pat in _KEY_MAP.items():
        if "{}" not in model_pat:
            if fqn == model_pat:
                return gguf_pat
            continue
        prefix, suffix = model_pat.split("{}")
        if fqn.startswith(prefix) and fqn.endswith(suffix):
            idx = fqn[len(prefix) : len(fqn) - len(suffix)]
            if idx.isdigit():
                return gguf_pat.replace("{}", idx)
    return None


def build_gguf_checkpoint(path: str, config: Gemma4_31BConfig = GGUF_CONFIG) -> None:
    """Write a tiny GGUF file matching ``config``.

    Linears are Q4_K; ``ffn_down`` / ``token_embd`` are Q6_K (to exercise both
    GGUF unpack paths); norms / scalars are F32. Tensor shapes are derived from
    the instantiated model so per-layer-type differences (e.g. global layers
    having no v_proj / q_norm) are handled automatically. ``output.weight`` is
    omitted -- GGUF ties lm_head to the token embedding. Requires the ``gguf``
    package.
    """
    import gguf
    from executorch.extension.llm.export.gguf import QK_K
    from executorch.extension.llm.export.test.test_gguf import (
        _make_q4k_raw,
        _make_q6k_raw,
    )

    with torch.device("meta"):
        model = Gemma4_31B(config)

    writer = gguf.GGUFWriter(path, "gemma")
    for fqn, p in model.named_parameters():
        gguf_key = _model_to_gguf_key(fqn)
        if gguf_key is None:
            continue
        if p.dim() == 2:
            N, K = int(p.shape[0]), int(p.shape[1])
            nb = K // QK_K
            use_q6 = gguf_key == "token_embd.weight" or gguf_key.endswith(
                "ffn_down.weight"
            )
            blob = (_make_q6k_raw(N, nb) if use_q6 else _make_q4k_raw(N, nb)).numpy()
            raw_dtype = (
                gguf.GGMLQuantizationType.Q6_K
                if use_q6
                else gguf.GGMLQuantizationType.Q4_K
            )
            writer.add_tensor(gguf_key, blob, raw_dtype=raw_dtype)
        else:
            arr = (torch.randn(tuple(p.shape), dtype=torch.float32) * 0.1).numpy()
            writer.add_tensor(gguf_key, arr)
    # Per-layer scalars are buffers (not parameters) but are stored in real
    # GGUFs (e.g. blk.N.layer_output_scale.weight). Write the ones that have a
    # GGUF mapping so they load as bf16; runtime buffers (RoPE, KV cache, ...)
    # map to None and are skipped.
    for fqn, b in model.named_buffers():
        gguf_key = _model_to_gguf_key(fqn)
        if gguf_key is None:
            continue
        arr = torch.ones(tuple(b.shape), dtype=torch.float32).numpy()
        writer.add_tensor(gguf_key, arr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ---------------------------------------------------------------------------
# Tests (CPU only, no backend dependency)


class TestQuantizeSaveLoadRoundtrip(unittest.TestCase):
    def test_roundtrip_preserves_weights(self):
        """quantize → save → load recovers all weights."""
        from torchao.quantization import IntxUnpackedToInt8Tensor
        from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        # Save form == quantize_stream (torchao-native Int4Tensor), which is what
        # quantize_and_save writes to disk; quantize_model's Exportable form is
        # in-memory only and would need torchao registration to flatten.
        params = ((fqn, p.data) for fqn, p in model.named_parameters())
        state_dict = dict(quantize_stream(params, DEFAULT_RECIPE, dtype=torch.bfloat16))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.safetensors")
            td, md = flatten_tensor_state_dict(state_dict)
            save_file(td, path, metadata=md)
            with safe_open(path, framework="pt", device="cpu") as f:
                loaded_meta = f.metadata()
                loaded_tensors = {k: f.get_tensor(k) for k in f.keys()}
            loaded, _ = unflatten_tensor_state_dict(loaded_tensors, loaded_meta)

        self.assertEqual(set(state_dict.keys()), set(loaded.keys()))
        for fqn in state_dict:
            orig = state_dict[fqn]
            got = loaded[fqn]
            self.assertEqual(type(orig).__name__, type(got).__name__)
            if isinstance(orig, Int4Tensor):
                self.assertTrue(torch.equal(orig.qdata, got.qdata))
                self.assertTrue(torch.equal(orig.scale, got.scale))
            elif isinstance(orig, IntxUnpackedToInt8Tensor):
                self.assertTrue(torch.equal(orig.qdata, got.qdata))
                self.assertTrue(torch.equal(orig.scale, got.scale))
            elif isinstance(orig, torch.Tensor):
                self.assertTrue(torch.equal(orig, got))

    def test_embedding_quantized_as_int8(self):
        """embed_tokens is quantized to INT8 (IntxUnpackedToInt8Tensor)."""
        from torchao.quantization import IntxUnpackedToInt8Tensor

        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        state_dict = quantize_model(model, DEFAULT_RECIPE)

        self.assertIn("embed_tokens.weight", state_dict)
        self.assertIsInstance(
            state_dict["embed_tokens.weight"], IntxUnpackedToInt8Tensor
        )

    def test_model_free_save_matches_quantize_model(self):
        """Model-free save (iter_checkpoint + tie_map + quantize_stream) reconstructs
        the model-based quantize_model state dict.

        quantize_stream yields the serialization form (torchao-native Int4Tensor);
        quantize_model yields the in-memory Exportable form. They match once the
        streamed output is passed through the same ``to_default`` convert.
        """
        from executorch.examples.models.gemma4_31b.model import _hf_to_model_key
        from executorch.extension.llm.export.load import iter_checkpoint
        from executorch.extension.llm.export.quant import identity

        # Reference: instantiate the model, untie lm_head, quantize_model.
        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        ref = quantize_model(model, DEFAULT_RECIPE)

        # Model-free: stream a tied HF checkpoint (same seed-42 weights), remap
        # names, fan lm_head out of the embedding (clone=True), quantize per recipe.
        with tempfile.TemporaryDirectory() as tmpdir:
            build_hf_checkpoint(tmpdir)
            pairs = iter_checkpoint(
                tmpdir,
                key_map=_hf_to_model_key,
                convert=identity,
                tie_map={"embed_tokens.weight": ("lm_head.weight", True)},
            )
            streamed = {
                fqn: to_default(fqn, v)
                for fqn, v in quantize_stream(
                    pairs, DEFAULT_RECIPE, dtype=torch.bfloat16
                )
            }

        self.assertEqual(set(streamed), set(ref))
        for fqn, r in ref.items():
            got = streamed[fqn]
            self.assertIs(type(got), type(r), fqn)
            data_names = getattr(r, "tensor_data_names", None)
            if data_names:  # quantized subclass -> compare payload bit-exactly
                for n in data_names:
                    self.assertTrue(torch.equal(getattr(got, n), getattr(r, n)), fqn)
            else:
                self.assertTrue(torch.equal(got, r), fqn)

    def test_load_casts_int4_to_fp16(self):
        """Loading a bf16-quantized checkpoint with dtype=fp16 re-stamps int4.

        Regression for the convert-then-cast order in iter_checkpoint: a raw
        torchao Int4Tensor (whose own .to ignores dtype) is first wrapped as an
        ExportableInt4Tensor by convert, then re-stamped to fp16 by maybe_cast.
        """
        from executorch.extension.llm.export.int4 import ExportableInt4Tensor
        from executorch.extension.llm.export.load import iter_checkpoint
        from executorch.extension.llm.export.quant import to_default
        from torchao.quantization import IntxUnpackedToInt8Tensor
        from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

        model = build_random_tiny_model()
        model.lm_head.weight = nn.Parameter(model.embed_tokens.weight.clone())
        # Save the torchao-native form (quantize_stream) -- the real on-disk form.
        params = ((fqn, p.data) for fqn, p in model.named_parameters())
        state_dict = dict(quantize_stream(params, DEFAULT_RECIPE, dtype=torch.bfloat16))
        int4_fqns = {f for f, v in state_dict.items() if isinstance(v, Int4Tensor)}
        int8_fqns = {
            f for f, v in state_dict.items() if isinstance(v, IntxUnpackedToInt8Tensor)
        }
        self.assertTrue(int4_fqns and int8_fqns)

        with tempfile.TemporaryDirectory() as tmpdir:
            td, md = flatten_tensor_state_dict(state_dict)
            save_file(td, os.path.join(tmpdir, "model.safetensors"), metadata=md)
            loaded = dict(
                iter_checkpoint(tmpdir, convert=to_default, dtype=torch.float16)
            )

        for fqn in int4_fqns:
            got = loaded[fqn]
            self.assertIsInstance(got, ExportableInt4Tensor, fqn)
            self.assertEqual(got.orig_dtype, torch.float16, fqn)
            self.assertEqual(got.scale.dtype, torch.float16, fqn)
            self.assertEqual(got.dequantize().dtype, torch.float16, fqn)
        for fqn in int8_fqns:
            got = loaded[fqn]
            self.assertIsInstance(got, IntxUnpackedToInt8Tensor, fqn)
            self.assertEqual(got.scale.dtype, torch.float16, fqn)


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


class TestGgufKeyMapping(unittest.TestCase):
    """Unit tests for gguf_loader.gguf_to_model_key (CPU, no GGUF file needed)."""

    def test_attention_keys(self):
        from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key

        self.assertEqual(
            gguf_to_model_key("blk.0.attn_q.weight"),
            "layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            gguf_to_model_key("blk.59.attn_output.weight"),
            "layers.59.self_attn.o_proj.weight",
        )

    def test_mlp_keys(self):
        from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key

        self.assertEqual(
            gguf_to_model_key("blk.5.ffn_gate.weight"),
            "layers.5.mlp.gate_proj.weight",
        )

    def test_global_keys(self):
        from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key

        self.assertEqual(gguf_to_model_key("token_embd.weight"), "embed_tokens.weight")
        self.assertEqual(gguf_to_model_key("output_norm.weight"), "norm.weight")

    def test_unknown_key_returns_none(self):
        from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key

        self.assertIsNone(gguf_to_model_key("blk.0.some_unknown.weight"))

    def test_ignored_key_returns_none(self):
        from executorch.examples.models.gemma4_31b.gguf_loader import gguf_to_model_key

        self.assertIsNone(gguf_to_model_key("rope_freqs.weight"))


if __name__ == "__main__":
    unittest.main()
