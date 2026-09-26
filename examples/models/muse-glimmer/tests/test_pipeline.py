# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic integration tests for the Muse Glimmer pipeline.

Tests the quantize -> save -> load roundtrip on a tiny model, KV cache
correctness, and checkpoint key mapping. No CUDA required. Backend-specific
tests (pack, inference, export) live in ``test_cuda_pipeline.py``.

Usage:
    python -m pytest examples/models/muse-glimmer/tests/test_pipeline.py -v
"""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass, replace

import torch
from executorch.examples.models.muse_glimmer.model.model import (
    _ckpt_to_model_key,
    _fix_raw_key,
    _split_fused_qkv,
    _split_fused_qkv_to_qko,
    FlatKVCache,
    MuseGlimmerConfig,
    MuseGlimmerModel,
    RingKVCache,
)

try:
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
    from executorch.backends.cuda.dp4a_planar_int6_tensor import (
        CudaDp4aPlanarInt6Tensor,
    )
    from executorch.examples.models.muse_glimmer.loaders.quantize_and_save import (
        build_recipes,
    )
    from executorch.extension.llm.export.quant import (
        identity,
        QuantConfig,
        quantize_model,
        QuantRecipe,
        QuantRule,
    )
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
        unflatten_tensor_state_dict,
    )
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    HAS_QUANT_DEPS = True
except Exception:
    # Any import-time failure (missing package, or a broken torchao/triton/torch
    # version skew that raises AttributeError) means the quant deps are unusable.
    HAS_QUANT_DEPS = False


# ---------------------------------------------------------------------------
# Shared fixtures — imported by test_cuda_pipeline.py.

TINY_CONFIG = MuseGlimmerConfig(
    dim=256,
    n_layers=8,
    n_heads=4,
    n_kv_heads=2,
    head_dim=64,
    ffn_dim_multiplier=2.0,
    vocab_size=256,
    norm_eps=1e-5,
    rope_theta=500_000.0,
    use_qk_norm=True,
    use_attn_o_gate=True,
    output_soft_cap_temp=20.0,
    global_attn_cfg="[16,16,16,0]",
    every_n_layers_nope=4,
    normalize_tok_embeddings=True,
    output_multiplier=0.19611613513,
    post_norm_eps=1e-8,
    max_seq_len=64,
)

# GGUF fixtures need K (= dim) a multiple of the k-quant super-block QK_K=256.
# head_dim*n_kv_heads and the fused dims stay small; only ``dim`` must be 256.
GGUF_CONFIG = MuseGlimmerConfig(
    dim=256,
    n_layers=8,
    n_heads=4,
    n_kv_heads=2,
    head_dim=64,
    ffn_dim_multiplier=1.0,
    vocab_size=256,
    norm_eps=1e-5,
    rope_theta=500_000.0,
    use_qk_norm=True,
    use_attn_o_gate=True,
    output_soft_cap_temp=20.0,
    global_attn_cfg="[16,16,16,0]",
    every_n_layers_nope=4,
    normalize_tok_embeddings=True,
    output_multiplier=0.19611613513,
    post_norm_eps=1e-8,
    max_seq_len=64,
)

if HAS_QUANT_DEPS:
    QUANT_4W = QuantConfig(bits=4, group_size=32, symmetric=False, method="min_max")
    QUANT_8W_PER_AXIS = QuantConfig(
        bits=8,
        group_size=TINY_CONFIG.dim,
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


@dataclass(frozen=True)
class MockEncoding:
    """The `.ids` carrier that tokenizers.Tokenizer.encode returns."""

    ids: list[int]


class MockTokenizer:
    """Mirrors the tokenizers.Tokenizer surface inference.py calls."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> MockEncoding:
        return MockEncoding([1, 2, 3, 4])

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "<tokens:" + ",".join(str(i) for i in ids) + ">"

    def token_to_id(self, token: str) -> int | None:
        return {"<|patch|>": 200092}.get(token)


def build_random_tiny_model() -> MuseGlimmerModel:
    torch.manual_seed(42)
    model = MuseGlimmerModel(TINY_CONFIG)
    model.to(dtype=torch.bfloat16)
    for p in model.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.02)
    model.eval()
    return model


def save_checkpoint(output_dir: str) -> None:
    """Build tiny model, quantize, and save as prequantized checkpoint."""
    model = build_random_tiny_model()
    state_dict = quantize_model(model, DEFAULT_RECIPE, convert=identity)
    os.makedirs(output_dir, exist_ok=True)
    td, md = flatten_tensor_state_dict(state_dict)
    save_file(td, os.path.join(output_dir, "model.safetensors"), metadata=md)

    config_dict = {
        "dim": TINY_CONFIG.dim,
        "n_layers": TINY_CONFIG.n_layers,
        "n_heads": TINY_CONFIG.n_heads,
        "n_kv_heads": TINY_CONFIG.n_kv_heads,
        "head_dim": TINY_CONFIG.head_dim,
        "ffn_dim_multiplier": TINY_CONFIG.ffn_dim_multiplier,
        "vocab_size": TINY_CONFIG.vocab_size,
        "norm_eps": TINY_CONFIG.norm_eps,
        "rope_theta": TINY_CONFIG.rope_theta,
        "use_qk_norm": TINY_CONFIG.use_qk_norm,
        "use_attn_o_gate": TINY_CONFIG.use_attn_o_gate,
        "output_soft_cap_temp": TINY_CONFIG.output_soft_cap_temp,
        "global_attn_cfg": TINY_CONFIG.global_attn_cfg,
        "every_n_layers_nope": TINY_CONFIG.every_n_layers_nope,
        "normalize_tok_embeddings": TINY_CONFIG.normalize_tok_embeddings,
        "output_multiplier": TINY_CONFIG.output_multiplier,
        "post_norm_eps": TINY_CONFIG.post_norm_eps,
    }
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(config_dict, f)


# ---------------------------------------------------------------------------
# Tests (CPU only, no backend dependency)


@unittest.skipUnless(HAS_QUANT_DEPS, "torchao quantization dependencies not available")
class TestQuantizeSaveLoadRoundtripTest(unittest.TestCase):
    def test_roundtrip_preserves_weights(self):
        """quantize -> save -> load recovers all weights."""
        model = build_random_tiny_model()
        state_dict = quantize_model(model, DEFAULT_RECIPE, convert=identity)

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
        model = build_random_tiny_model()
        state_dict = quantize_model(model, DEFAULT_RECIPE, convert=identity)

        self.assertIn("embed_tokens.weight", state_dict)
        self.assertIsInstance(
            state_dict["embed_tokens.weight"], IntxUnpackedToInt8Tensor
        )

    def test_norms_not_quantized(self):
        """Norm weights stay as plain tensors (not quantized)."""
        model = build_random_tiny_model()
        state_dict = quantize_model(model, DEFAULT_RECIPE, convert=identity)

        norm_keys = [k for k in state_dict if "norm" in k]
        self.assertGreater(len(norm_keys), 0)
        for k in norm_keys:
            self.assertIsInstance(state_dict[k], torch.Tensor)
            self.assertNotIn("Int4", type(state_dict[k]).__name__)


class TestRingKVCacheTest(unittest.TestCase):
    def _make_cache(self, window=4, heads=2, head_dim=8):
        return RingKVCache(
            max_batch_size=1,
            window_size=window,
            num_kv_heads=heads,
            head_dim=head_dim,
        )

    def test_sequential_write_read(self):
        """Writing positions 0..buf_size-1 fills every slot exactly once."""
        cache = self._make_cache(window=4)
        buf_size = cache.buf_size
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
        buf_size = cache.buf_size
        for i in range(buf_size + 1):
            pos = torch.tensor([i], dtype=torch.long)
            k = torch.full((1, 2, 1, 8), float(i))
            v = torch.full((1, 2, 1, 8), float(i))
            k_out, _ = cache.update(pos, k, v)
        self.assertEqual(k_out[0, 0, 0, 0].item(), float(buf_size))
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


class TestFlatKVCacheTest(unittest.TestCase):
    def _make_cache(self, max_seq=16, heads=2, head_dim=8):
        return FlatKVCache(
            max_batch_size=1,
            max_seq_len=max_seq,
            num_kv_heads=heads,
            head_dim=head_dim,
        )

    def test_sequential_write_read(self):
        cache = self._make_cache()
        for i in range(8):
            pos = torch.tensor([i], dtype=torch.long)
            k = torch.full((1, 2, 1, 8), float(i))
            v = torch.full((1, 2, 1, 8), float(i + 100))
            k_out, v_out = cache.update(pos, k, v)
        for i in range(8):
            self.assertEqual(k_out[0, 0, i, 0].item(), float(i))
            self.assertEqual(v_out[0, 0, i, 0].item(), float(i + 100))

    def test_multi_token_prefill(self):
        cache = self._make_cache()
        pos = torch.arange(6, dtype=torch.long)
        k = torch.arange(6).float().view(1, 1, 6, 1).expand(1, 2, 6, 8)
        v = torch.zeros(1, 2, 6, 8)
        k_out, _ = cache.update(pos, k, v)
        for i in range(6):
            self.assertEqual(k_out[0, 0, i, 0].item(), float(i))


class TestCheckpointKeyMappingTest(unittest.TestCase):
    def test_raw_key_fixes(self):
        self.assertEqual(
            _fix_raw_key("layers.0.attention.qkv_proj.norm_weight"),
            "layers.0.attention.qkv_proj_norm.weight",
        )
        self.assertEqual(
            _fix_raw_key("layers.5.feed_forward.norm_weight"),
            "layers.5.feed_forward.norm.weight",
        )
        self.assertEqual(
            _fix_raw_key("output.norm_weight"),
            "output_norm.weight",
        )
        self.assertEqual(
            _fix_raw_key("output.norm.weight"),
            "output_norm.weight",
        )

    def test_attention_keys(self):
        # qkv_proj_norm and o_proj map directly.
        self.assertEqual(
            _ckpt_to_model_key("layers.0.attention.qkv_proj_norm.weight"),
            "layers.0.self_attn.qkv_proj_norm.weight",
        )
        self.assertEqual(
            _ckpt_to_model_key("layers.51.attention.o_proj.weight"),
            "layers.51.self_attn.o_proj.weight",
        )
        # The fused qkv_proj weight maps directly to the model's fused qkv_proj;
        # the split/qko layouts are produced by a post-load split in
        # from_checkpoint (_split_fused_qkv / _split_fused_qkv_to_qko).
        self.assertEqual(
            _ckpt_to_model_key("layers.3.attention.qkv_proj.weight"),
            "layers.3.self_attn.qkv_proj.weight",
        )

    def test_fused_qkv_split_to_qko(self):
        """The consolidated [Q|K|V|OG] weight splits into qko_proj + v_proj."""
        cfg = TINY_CONFIG
        q_dim = cfg.n_heads * cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        og_dim = q_dim if cfg.use_attn_o_gate else 0
        fused = torch.randn(q_dim + 2 * kv_dim + og_dim, cfg.dim)
        prefix = "layers.7.self_attn."
        sd = {f"{prefix}qkv_proj.weight": fused}

        out = _split_fused_qkv_to_qko(sd, cfg)
        qko = out[f"{prefix}qko_proj.weight"]
        v = out[f"{prefix}v_proj.weight"]

        # Shapes: qko = [Q|K|OG], v = [V].
        self.assertEqual(qko.shape, (q_dim + kv_dim + og_dim, cfg.dim))
        self.assertEqual(v.shape, (kv_dim, cfg.dim))
        self.assertNotIn(f"{prefix}qkv_proj.weight", out)

        # Row-for-row content: qko is Q, K, then OG; v is the middle V slice.
        self.assertTrue(torch.equal(qko[:q_dim], fused[:q_dim]))
        self.assertTrue(
            torch.equal(qko[q_dim : q_dim + kv_dim], fused[q_dim : q_dim + kv_dim])
        )
        self.assertTrue(torch.equal(qko[q_dim + kv_dim :], fused[q_dim + 2 * kv_dim :]))
        self.assertTrue(torch.equal(v, fused[q_dim + kv_dim : q_dim + 2 * kv_dim]))

    def test_mlp_keys(self):
        self.assertEqual(
            _ckpt_to_model_key("layers.0.feed_forward.norm.weight"),
            "layers.0.mlp.norm.weight",
        )
        self.assertEqual(
            _ckpt_to_model_key("layers.5.feed_forward.fc1_weight"),
            "layers.5.mlp.gate_up_proj.weight",
        )
        self.assertEqual(
            _ckpt_to_model_key("layers.5.feed_forward.fc2_weight"),
            "layers.5.mlp.down_proj.weight",
        )

    def test_global_keys(self):
        self.assertEqual(
            _ckpt_to_model_key("tok_embeddings.weight"),
            "embed_tokens.weight",
        )
        self.assertEqual(
            _ckpt_to_model_key("output_norm.weight"),
            "output_norm.weight",
        )
        self.assertEqual(
            _ckpt_to_model_key("output.weight"),
            "lm_head.weight",
        )

    def test_post_norm_keys_pass_through(self):
        self.assertEqual(
            _ckpt_to_model_key("layers.0.post_attn_norm.weight"),
            "layers.0.post_attn_norm.weight",
        )
        self.assertEqual(
            _ckpt_to_model_key("layers.0.post_ffn_norm.weight"),
            "layers.0.post_ffn_norm.weight",
        )

    def test_vision_keys_ignored(self):
        self.assertIsNone(_ckpt_to_model_key("vision_projection.weight"))
        self.assertIsNone(_ckpt_to_model_key("perception_emb_norm.weight"))

    def test_unknown_key_returns_none(self):
        self.assertIsNone(_ckpt_to_model_key("some.unknown.key"))


class TestLayerClassificationTest(unittest.TestCase):
    def test_tiny_config_layer_types(self):
        """Verify iRoPE and SWA layer classification on tiny config."""
        cfg = TINY_CONFIG
        rope_layers = [i for i in range(cfg.n_layers) if cfg.layer_use_rope(i)]
        nope_layers = [i for i in range(cfg.n_layers) if not cfg.layer_use_rope(i)]
        swa_layers = [i for i in range(cfg.n_layers) if cfg.layer_window_size(i) > 0]
        global_layers = [
            i for i in range(cfg.n_layers) if cfg.layer_window_size(i) == 0
        ]

        self.assertEqual(nope_layers, global_layers)
        self.assertEqual(rope_layers, swa_layers)
        self.assertEqual(len(rope_layers) + len(nope_layers), cfg.n_layers)

    def test_full_config_layer_types(self):
        """Verify 39 RoPE+SWA and 13 NoPE+Global on the real 52-layer config."""
        cfg = MuseGlimmerConfig()
        rope_count = sum(1 for i in range(cfg.n_layers) if cfg.layer_use_rope(i))
        global_count = sum(
            1 for i in range(cfg.n_layers) if cfg.layer_window_size(i) == 0
        )
        self.assertEqual(rope_count, 39)
        self.assertEqual(global_count, 13)


class TestTinyModelForwardTest(unittest.TestCase):
    def test_forward_produces_logits(self):
        """Tiny model forward pass produces valid logits on CPU."""
        model = build_random_tiny_model()
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        input_pos = torch.arange(3, dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens, input_pos)
        self.assertEqual(logits.shape, (1, 3, TINY_CONFIG.vocab_size))
        self.assertFalse(logits.isnan().any())
        self.assertFalse(logits.isinf().any())

    def test_decode_step_produces_logits(self):
        """Single-token decode step produces valid logits."""
        model = build_random_tiny_model()
        tokens = torch.tensor([[5]], dtype=torch.long)
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens, input_pos)
        self.assertEqual(logits.shape, (1, 1, TINY_CONFIG.vocab_size))
        self.assertFalse(logits.isnan().any())

    def test_soft_capping_bounds_logits(self):
        """Output soft-capping keeps logits within [-cap, cap]."""
        model = build_random_tiny_model()
        tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        input_pos = torch.arange(5, dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens, input_pos)
        cap = TINY_CONFIG.output_soft_cap_temp
        self.assertTrue((logits.abs() <= cap + 1e-5).all())


class TestUnfuseQkvTest(unittest.TestCase):
    def test_split_fused_qkv_shapes_and_values(self):
        """_split_fused_qkv splits the fused rows into q/k/v/og by size."""
        cfg = replace(TINY_CONFIG, fuse_qkv=False)
        q_dim = cfg.n_heads * cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        og_dim = q_dim
        fused_w = torch.randn(q_dim + 2 * kv_dim + og_dim, cfg.dim)
        prefix = "layers.0.self_attn."
        sd = {
            f"{prefix}qkv_proj.weight": fused_w,
            "embed_tokens.weight": torch.randn(4, cfg.dim),
        }

        out = _split_fused_qkv(sd, cfg)

        self.assertNotIn(f"{prefix}qkv_proj.weight", out)
        self.assertIn("embed_tokens.weight", out)
        self.assertEqual(out[f"{prefix}q_proj.weight"].shape, (q_dim, cfg.dim))
        self.assertEqual(out[f"{prefix}k_proj.weight"].shape, (kv_dim, cfg.dim))
        self.assertEqual(out[f"{prefix}v_proj.weight"].shape, (kv_dim, cfg.dim))
        self.assertEqual(out[f"{prefix}og_proj.weight"].shape, (og_dim, cfg.dim))
        self.assertTrue(
            torch.equal(
                out[f"{prefix}v_proj.weight"],
                fused_w[q_dim + kv_dim : q_dim + 2 * kv_dim],
            )
        )

    def test_fused_and_unfused_forward_match(self):
        """Unfused q/k/v/og projections reproduce the fused forward output."""
        torch.manual_seed(0)
        fused = MuseGlimmerModel(replace(TINY_CONFIG, fuse_qkv=True)).to(torch.bfloat16)
        for p in fused.parameters():
            p.data.normal_(0, 0.02)
        fused.eval()

        unfused_cfg = replace(TINY_CONFIG, fuse_qkv=False)
        unfused = MuseGlimmerModel(unfused_cfg).to(torch.bfloat16)
        unfused.load_state_dict(
            _split_fused_qkv(fused.state_dict(), unfused_cfg), strict=False
        )
        unfused.eval()

        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        input_pos = torch.arange(4, dtype=torch.long)
        with torch.no_grad():
            out_fused = fused(tokens, input_pos)
            out_unfused = unfused(tokens, input_pos)

        torch.testing.assert_close(out_fused, out_unfused, atol=1e-1, rtol=1e-1)


@unittest.skipUnless(HAS_QUANT_DEPS, "torchao quantization dependencies not available")
class TestGgufRecipeTest(unittest.TestCase):
    def test_gguf_bit_assignment(self):
        """gguf recipe: int6 on v_proj/down_proj/lm_head, int4 elsewhere."""
        recipe = build_recipes()["gguf"]
        for fqn in (
            "layers.0.self_attn.v_proj.weight",
            "layers.7.mlp.down_proj.weight",
            "lm_head.weight",
        ):
            cfg = recipe.get_config(fqn)
            self.assertIsNotNone(cfg, fqn)
            self.assertEqual(cfg.bits, 6, fqn)
            self.assertEqual(cfg.group_size, 32, fqn)
        for fqn in (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.og_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.mlp.gate_up_proj.weight",
            "embed_tokens.weight",
        ):
            cfg = recipe.get_config(fqn)
            self.assertIsNotNone(cfg, fqn)
            self.assertEqual(cfg.bits, 4, fqn)
        self.assertIsNone(recipe.get_config("layers.0.post_attn_norm.weight"))

    def test_gguf_int8_bit_assignment(self):
        """gguf_int8 recipe: int8 gs128 on v_proj/down_proj/lm_head, int4 else."""
        recipe = build_recipes()["gguf_int8"]
        for fqn in (
            "layers.0.self_attn.v_proj.weight",
            "layers.7.mlp.down_proj.weight",
            "lm_head.weight",
        ):
            cfg = recipe.get_config(fqn)
            self.assertIsNotNone(cfg, fqn)
            self.assertEqual(cfg.bits, 8, fqn)
            self.assertEqual(cfg.group_size, 128, fqn)
        for fqn in (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.og_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.mlp.gate_up_proj.weight",
            "embed_tokens.weight",
        ):
            cfg = recipe.get_config(fqn)
            self.assertIsNotNone(cfg, fqn)
            self.assertEqual(cfg.bits, 4, fqn)
        self.assertIsNone(recipe.get_config("layers.0.post_attn_norm.weight"))

    def test_int4_recipe_all_int4(self):
        """int4 recipe: int4 everywhere incl. embedding; norms unquantized."""
        recipe = build_recipes()["int4"]
        for fqn in (
            "embed_tokens.weight",
            "layers.0.self_attn.qkv_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.mlp.gate_up_proj.weight",
            "layers.7.mlp.down_proj.weight",
            "lm_head.weight",
        ):
            cfg = recipe.get_config(fqn)
            self.assertIsNotNone(cfg, fqn)
            self.assertEqual(cfg.bits, 4, fqn)
        self.assertIsNone(recipe.get_config("layers.0.post_attn_norm.weight"))


# ---------------------------------------------------------------------------
# GGUF native-bit-width loading (CPU: packing needs no CUDA)


# Q6_K shards per GGUF leaf: which layers store this leaf at Q6_K (else Q4_K).
# Exercises mixed per-layer bit-width for the standalone v_proj and ffn_down.
_GGUF_Q6_LAYERS = {
    "attn_v": {0, 3, 5},
    "ffn_down": {1, 4},
}
# Top-level output (lm_head) stored at Q6_K.
_GGUF_Q6_OUTPUT = True


def build_muse_glimmer_gguf(path: str, config: MuseGlimmerConfig = GGUF_CONFIG) -> None:
    """Write a tiny Muse Glimmer GGUF with mixed Q4_K/Q6_K to exercise bit-width routing.

    Layout mirrors a real Muse Glimmer GGUF: separate ``attn_q/k/v/output_gate`` and
    ``ffn_gate/up`` shards (fused only inside the model), plus ``ffn_down``,
    ``attn_output``, norms, ``token_embd`` (Q4_K), and ``output`` (Q6_K).
    ``attn_v`` / ``ffn_down`` are Q6_K on the layers in ``_GGUF_Q6_LAYERS``.
    Requires the ``gguf`` package. K (= dim) must be a multiple of QK_K.
    """
    import gguf
    from executorch.extension.llm.export.gguf import (
        _Q4_K_BLOCK_BYTES,
        _Q6_K_BLOCK_BYTES,
        QK_K,
    )

    def _fp16_bytes(x: float):
        return torch.tensor([x], dtype=torch.float16).view(torch.uint8)

    def _make_q4k_raw(n: int, nb: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        blk = torch.randint(
            0, 256, (n * nb, _Q4_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
        )
        blk[:, 0:2] = _fp16_bytes(0.01)  # d
        blk[:, 2:4] = _fp16_bytes(0.01)  # dmin
        blk[:, 4:16] = 0x21  # fixed mid-range 6-bit sub-scales/mins (non-zero)
        return blk.reshape(n, nb * _Q4_K_BLOCK_BYTES)

    def _make_q6k_raw(n: int, nb: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        blk = torch.randint(
            0, 256, (n * nb, _Q6_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
        )
        blk[:, 192:208] = 0x10  # fixed int8 sub-scales (non-zero)
        blk[:, 208:210] = _fp16_bytes(0.01)  # d
        return blk.reshape(n, nb * _Q6_K_BLOCK_BYTES)

    dim = config.dim
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    ffn = config.ffn_dim
    nb = dim // QK_K  # blocks along K (all linears project from `dim`)

    def q4(n: int, seed: int):
        return _make_q4k_raw(n, nb, seed).numpy(), gguf.GGMLQuantizationType.Q4_K

    def q6(n: int, seed: int):
        return _make_q6k_raw(n, nb, seed).numpy(), gguf.GGMLQuantizationType.Q6_K

    def f32(*shape):
        return torch.ones(shape, dtype=torch.float32).numpy()

    writer = gguf.GGUFWriter(path, "muse_glimmer")
    seed = 0
    for i in range(config.n_layers):
        # Fused-in-model shards (always Q4_K).
        for leaf, n in (
            ("attn_q", q_dim),
            ("attn_k", kv_dim),
            ("attn_output_gate", q_dim),
            ("ffn_gate", ffn),
            ("ffn_up", ffn),
        ):
            blob, dt = q4(n, seed)
            seed += 1
            writer.add_tensor(f"blk.{i}.{leaf}.weight", blob, raw_dtype=dt)
        # Standalone shards with per-layer bit-width.
        for leaf, n in (("attn_v", kv_dim), ("ffn_down", dim), ("attn_output", dim)):
            use_q6 = i in _GGUF_Q6_LAYERS.get(leaf, set())
            blob, dt = q6(n, seed) if use_q6 else q4(n, seed)
            seed += 1
            writer.add_tensor(f"blk.{i}.{leaf}.weight", blob, raw_dtype=dt)
        # Norms (F32) + all-ones QK-norms (ignored by the loader).
        writer.add_tensor(f"blk.{i}.attn_norm.weight", f32(dim))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", f32(dim))
        writer.add_tensor(f"blk.{i}.post_attention_norm.weight", f32(dim))
        writer.add_tensor(f"blk.{i}.post_ffw_norm.weight", f32(dim))
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", f32(config.head_dim))
        writer.add_tensor(f"blk.{i}.attn_k_norm.weight", f32(config.head_dim))

    # Top-level: token_embd (Q4_K), output/lm_head (Q6_K), output_norm (F32).
    emb, dt = q4(config.vocab_size, seed)
    seed += 1
    writer.add_tensor("token_embd.weight", emb, raw_dtype=dt)
    out, dt = (
        q6(config.vocab_size, seed) if _GGUF_Q6_OUTPUT else q4(config.vocab_size, seed)
    )
    seed += 1
    writer.add_tensor("output.weight", out, raw_dtype=dt)
    writer.add_tensor("output_norm.weight", f32(dim))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@unittest.skipUnless(HAS_QUANT_DEPS, "torchao quantization dependencies not available")
class TestGgufBitWidthTest(unittest.TestCase):
    """GGUF load preserves native per-tensor bit-width (Q4_K->INT4, Q6_K->INT6).

    Packing is CPU-safe (no CUDA required); only kernel execution needs CUDA.
    """

    def setUp(self):
        try:
            import gguf  # noqa: F401
        except ImportError:
            self.skipTest("gguf package required")

    def _load(self):
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_gguf_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.gguf")
            build_muse_glimmer_gguf(path)
            return load_gguf_model(path, backend="cuda", config=GGUF_CONFIG)

    def test_bit_widths_match_source(self):
        model, _ = self._load()

        # v_proj / ffn_down honor per-layer bit-width.
        for i in range(GGUF_CONFIG.n_layers):
            v = model.layers[i].self_attn.v_proj.weight.data
            want_v6 = i in _GGUF_Q6_LAYERS["attn_v"]
            self.assertIsInstance(
                v,
                CudaDp4aPlanarInt6Tensor if want_v6 else CudaCoalescedInt4Tensor,
                f"layer {i} v_proj bit-width mismatch",
            )
            d = model.layers[i].mlp.down_proj.weight.data
            want_d6 = i in _GGUF_Q6_LAYERS["ffn_down"]
            self.assertIsInstance(
                d,
                CudaDp4aPlanarInt6Tensor if want_d6 else CudaCoalescedInt4Tensor,
                f"layer {i} down_proj bit-width mismatch",
            )

        # Fused qko_proj / gate_up_proj (all-Q4_K shards) -> INT4.
        self.assertIsInstance(
            model.layers[0].self_attn.qko_proj.weight.data, CudaCoalescedInt4Tensor
        )
        self.assertIsInstance(
            model.layers[0].mlp.gate_up_proj.weight.data, CudaCoalescedInt4Tensor
        )
        # o_proj is Q4_K -> INT4.
        self.assertIsInstance(
            model.layers[0].self_attn.o_proj.weight.data, CudaCoalescedInt4Tensor
        )
        # lm_head is Q6_K -> INT6; token embedding is gatherable int4-in-int8.
        self.assertIsInstance(model.lm_head.weight.data, CudaDp4aPlanarInt6Tensor)
        self.assertIsInstance(model.embed_tokens.weight.data, IntxUnpackedToInt8Tensor)

    def test_fused_qko_shape(self):
        """qko_proj fuses [Q|K|OG]; v_proj is separate with [V] rows."""
        model, _ = self._load()
        q_dim = GGUF_CONFIG.n_heads * GGUF_CONFIG.head_dim
        kv_dim = GGUF_CONFIG.n_kv_heads * GGUF_CONFIG.head_dim
        og_dim = q_dim  # use_attn_o_gate
        self.assertEqual(
            model.layers[0].self_attn.qko_proj.weight.shape,
            (q_dim + kv_dim + og_dim, GGUF_CONFIG.dim),
        )
        self.assertEqual(
            model.layers[0].self_attn.v_proj.weight.shape,
            (kv_dim, GGUF_CONFIG.dim),
        )

    def test_mixed_quant_fusion_groups_remain_atomic(self):
        """A mixed-precision group stays separate across the whole model."""
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            _fuse_state_dict,
        )
        from executorch.extension.llm.export.gguf import (
            _Q4_K_BLOCK_BYTES,
            _Q6_K_BLOCK_BYTES,
            ExportableGGUFTensor,
        )

        def weight(rows, ggml_type, row_bytes):
            raw = torch.zeros((rows, row_bytes), dtype=torch.uint8)
            return ExportableGGUFTensor(raw, ggml_type, torch.bfloat16)

        atomic = {
            "layers.0.self_attn.q_proj.weight": weight(4, "q4_k", _Q4_K_BLOCK_BYTES),
            "layers.0.self_attn.k_proj.weight": weight(2, "q6_k", _Q6_K_BLOCK_BYTES),
            "layers.0.self_attn.og_proj.weight": weight(4, "q4_k", _Q4_K_BLOCK_BYTES),
            "layers.0.mlp.gate_proj.weight": weight(8, "q4_k", _Q4_K_BLOCK_BYTES),
            "layers.0.mlp.up_proj.weight": weight(8, "q6_k", _Q6_K_BLOCK_BYTES),
        }

        state_dict, fused_groups = _fuse_state_dict(atomic, "cuda", n_layers=1)

        self.assertEqual(fused_groups, set())
        self.assertEqual(set(state_dict), set(atomic))
        for fqn, tensor in atomic.items():
            self.assertIs(state_dict[fqn], tensor)


# ---------------------------------------------------------------------------
# mmproj (vision encoder) native-bit-width loading (CPU: packing needs no CUDA)


# Vision config small enough for a fast test; K (= latent_dim / adapter dims)
# must be a multiple of QK_K=256 for the k-quant block layout.
_MMPROJ_LATENT = 256
_MMPROJ_MLP = 512
_MMPROJ_ADAPTER = 256
_MMPROJ_HIDDEN = 256
_MMPROJ_ENC_OUT = _MMPROJ_LATENT * 4  # latent * downsample^2
_MMPROJ_LAYERS = 3
_MMPROJ_POS_TOKENS = 1024  # 32*32
_MMPROJ_PATCH_DIM = 2 * 3 * 14 * 14  # 1176


def _mmproj_vision_config():
    from executorch.examples.models.muse_glimmer.vision.precompute import (
        MuseGlimmerVisionConfig,
    )

    return MuseGlimmerVisionConfig(
        latent_dim=_MMPROJ_LATENT,
        n_heads=4,
        n_layers=_MMPROJ_LAYERS,
        mlp_hidden=_MMPROJ_MLP,
        adapter_dim=_MMPROJ_ADAPTER,
        encoder_output_dim=_MMPROJ_ENC_OUT,
        hidden_size=_MMPROJ_HIDDEN,
    )


def build_mmproj_gguf(
    path: str,
    *,
    numeric_projector_names: bool = False,
    folded_patch_embedding: bool = False,
) -> None:
    """Write a tiny mmproj (clip/muse_glimmer) GGUF exercising the vision name map.

    Mirrors the real schema: ``v.blk.{i}.{attn_q/k/out,ffn_up}`` Q4_K,
    ``attn_v``/``ffn_down`` Q6_K, all ``.bias``/norms F32, ``v.patch_embd`` and
    ``v.position_embd`` BF16, and ``mm.adapter_fc/adapter_proj/vision_proj``
    Q4_K. K dims are multiples of QK_K.
    """
    import gguf
    from executorch.extension.llm.export.gguf import (
        _Q4_K_BLOCK_BYTES,
        _Q6_K_BLOCK_BYTES,
        QK_K,
    )

    def _fp16_bytes(x: float):
        return torch.tensor([x], dtype=torch.float16).view(torch.uint8)

    def _q4k(n: int, k: int, seed: int):
        nb = k // QK_K
        g = torch.Generator().manual_seed(seed)
        blk = torch.randint(
            0, 256, (n * nb, _Q4_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
        )
        blk[:, 0:2] = _fp16_bytes(0.01)
        blk[:, 2:4] = _fp16_bytes(0.01)
        blk[:, 4:16] = 0x21
        return blk.reshape(n, nb * _Q4_K_BLOCK_BYTES).numpy(), (
            gguf.GGMLQuantizationType.Q4_K
        )

    def _q6k(n: int, k: int, seed: int):
        nb = k // QK_K
        g = torch.Generator().manual_seed(seed)
        blk = torch.randint(
            0, 256, (n * nb, _Q6_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
        )
        blk[:, 192:208] = 0x10
        blk[:, 208:210] = _fp16_bytes(0.01)
        return blk.reshape(n, nb * _Q6_K_BLOCK_BYTES).numpy(), (
            gguf.GGMLQuantizationType.Q6_K
        )

    def _f32(*shape):
        return torch.ones(shape, dtype=torch.float32).numpy()

    latent = _MMPROJ_LATENT
    writer = gguf.GGUFWriter(path, "clip")
    seed = 0
    for i in range(_MMPROJ_LAYERS):
        # Q4_K weights: attn_q/k/out (K=latent), ffn_up (K=latent).
        for leaf, n, k in (
            ("attn_q", latent, latent),
            ("attn_k", latent, latent),
            ("attn_out", latent, latent),
            ("ffn_up", _MMPROJ_MLP, latent),
        ):
            blob, dt = _q4k(n, k, seed)
            seed += 1
            writer.add_tensor(f"v.blk.{i}.{leaf}.weight", blob, raw_dtype=dt)
        # attn_v (Q6_K, K=latent), ffn_down (Q6_K, K=mlp).
        blob, dt = _q6k(latent, latent, seed)
        seed += 1
        writer.add_tensor(f"v.blk.{i}.attn_v.weight", blob, raw_dtype=dt)
        blob, dt = _q6k(latent, _MMPROJ_MLP, seed)
        seed += 1
        writer.add_tensor(f"v.blk.{i}.ffn_down.weight", blob, raw_dtype=dt)
        # Biases (F32) for each linear.
        for leaf, n in (
            ("attn_q", latent),
            ("attn_k", latent),
            ("attn_v", latent),
            ("attn_out", latent),
            ("ffn_up", _MMPROJ_MLP),
            ("ffn_down", latent),
        ):
            writer.add_tensor(f"v.blk.{i}.{leaf}.bias", _f32(n))
        # LayerNorms (weight + bias, F32).
        for leaf in ("ln1", "ln2"):
            writer.add_tensor(f"v.blk.{i}.{leaf}.weight", _f32(latent))
            writer.add_tensor(f"v.blk.{i}.{leaf}.bias", _f32(latent))

    # Patch/pos embed (F32 here; real mmproj uses BF16 — the loader casts both
    # float dtypes to bf16 identically). conv1_linear.weight is [latent,
    # patch_dim] (out, in); the positional table is [pos_tokens, latent] (used
    # by host grid_sample). The gguf writer + iter_gguf round-trip preserves the
    # numpy shape, so pass the natural torch shapes directly.
    if folded_patch_embedding:
        writer.add_tensor("v.patch_embd.weight", _f32(latent, 3, 14, 14))
    else:
        writer.add_tensor("v.patch_embd.weight", _f32(latent, _MMPROJ_PATCH_DIM))
    writer.add_tensor("v.position_embd.weight", _f32(_MMPROJ_POS_TOKENS, latent))
    writer.add_tensor("v.pre_ln.weight", _f32(latent))
    writer.add_tensor("v.pre_ln.bias", _f32(latent))
    writer.add_tensor("v.post_ln.weight", _f32(latent))
    writer.add_tensor("v.post_ln.bias", _f32(latent))

    # Adapter + projection (Q4_K). _q4k writes (out, in); adapter_fc maps
    # encoder_output_dim -> adapter_dim, so its weight is (adapter, enc_out).
    blob, dt = _q4k(_MMPROJ_ADAPTER, _MMPROJ_ENC_OUT, seed)
    seed += 1
    projector_names = (
        ("mm.0.weight", "mm.1.weight", "mm.2.weight")
        if numeric_projector_names
        else (
            "mm.adapter_fc.weight",
            "mm.adapter_proj.weight",
            "mm.vision_proj.weight",
        )
    )
    writer.add_tensor(projector_names[0], blob, raw_dtype=dt)
    blob, dt = _q4k(_MMPROJ_ADAPTER, _MMPROJ_ADAPTER, seed)
    seed += 1
    writer.add_tensor(projector_names[1], blob, raw_dtype=dt)
    blob, dt = _q4k(_MMPROJ_HIDDEN, _MMPROJ_ADAPTER, seed)
    seed += 1
    writer.add_tensor(projector_names[2], blob, raw_dtype=dt)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@unittest.skipUnless(HAS_QUANT_DEPS, "torchao quantization dependencies not available")
class TestMmprojLoadingTest(unittest.TestCase):
    """mmproj load packs vision weights at native bit-width; pos table returned."""

    def setUp(self):
        try:
            import gguf  # noqa: F401
        except ImportError:
            self.skipTest("gguf package required")

    def _load(self):
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mmproj_vision_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mmproj.gguf")
            build_mmproj_gguf(path)
            return load_mmproj_vision_model(path, config=_mmproj_vision_config())

    def test_bit_widths_and_pos_table(self):
        model, pos_table, cfg = self._load()

        # Q4_K attn/ffn_up/adapter -> INT4; Q6_K attn_v/ffn_down -> INT6.
        blk0 = model.blocks[0]
        self.assertIsInstance(blk0.attn.q_proj.weight.data, CudaCoalescedInt4Tensor)
        self.assertIsInstance(blk0.attn.o_proj.weight.data, CudaCoalescedInt4Tensor)
        self.assertIsInstance(blk0.attn.v_proj.weight.data, CudaDp4aPlanarInt6Tensor)
        self.assertIsInstance(blk0.mlp.c_fc.weight.data, CudaCoalescedInt4Tensor)
        self.assertIsInstance(blk0.mlp.c_proj.weight.data, CudaDp4aPlanarInt6Tensor)
        self.assertIsInstance(model.vision_proj.weight.data, CudaCoalescedInt4Tensor)

        # Biases + norms are plain bf16 tensors.
        self.assertEqual(blk0.attn.q_proj.bias.dtype, torch.bfloat16)
        self.assertEqual(blk0.ln_1.weight.dtype, torch.bfloat16)
        self.assertEqual(model.ln_pre.bias.dtype, torch.bfloat16)

        # conv1_linear (patch embed) is plain bf16.
        self.assertEqual(model.conv1_linear.weight.dtype, torch.bfloat16)

        # Positional table returned separately at [pos_tokens, latent].
        self.assertEqual(tuple(pos_table.shape), (_MMPROJ_POS_TOKENS, _MMPROJ_LATENT))
        self.assertEqual(pos_table.dtype, torch.bfloat16)

    def test_no_meta_params_left(self):
        model, _, _ = self._load()
        for fqn, p in model.named_parameters():
            self.assertNotEqual(p.device.type, "meta", f"{fqn} left on meta device")

    def test_numeric_projector_names_load_on_cuda_and_mlx(self):
        """The 30B numeric mmproj spelling is shared by both backends."""
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_mmproj_vision_model,
        )

        for backend in ("cuda", "mlx"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "mmproj-numeric.gguf")
                build_mmproj_gguf(
                    path,
                    numeric_projector_names=True,
                    folded_patch_embedding=True,
                )
                model, pos_table, _ = load_mmproj_vision_model(
                    path, config=_mmproj_vision_config(), backend=backend
                )
                self.assertEqual(
                    tuple(model.adapter_fc.weight.shape),
                    (_MMPROJ_ADAPTER, _MMPROJ_ENC_OUT),
                )
                self.assertEqual(
                    tuple(model.adapter_proj.weight.shape),
                    (_MMPROJ_ADAPTER, _MMPROJ_ADAPTER),
                )
                self.assertEqual(
                    tuple(model.vision_proj.weight.shape),
                    (_MMPROJ_HIDDEN, _MMPROJ_ADAPTER),
                )
                self.assertEqual(
                    tuple(pos_table.shape),
                    (_MMPROJ_POS_TOKENS, _MMPROJ_LATENT),
                )
                patch_weight = model.conv1_linear.weight.float()
                folded_dim = 3 * 14 * 14
                self.assertTrue(
                    torch.equal(
                        patch_weight[:, :folded_dim],
                        torch.ones_like(patch_weight[:, :folded_dim]),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        patch_weight[:, folded_dim:],
                        torch.zeros_like(patch_weight[:, folded_dim:]),
                    )
                )


def build_muse_glimmer_gguf_bf16(
    path: str, config: MuseGlimmerConfig = GGUF_CONFIG
) -> None:
    """Write a tiny all-bf16 Muse Glimmer GGUF (``general.file_type`` MOSTLY_BF16).

    Every weight is BF16 and every norm F32, matching a non-quantized GGUF
    (e.g. the muse_glimmer-provided reference). ``iter_gguf`` returns these as plain
    tensors, so the loader must skip int4 packing — the fused ``qko_proj`` /
    ``gate_up_proj`` groups are row-concatenated as bf16. Requires ``gguf``.
    """
    import gguf
    import numpy as np

    def bf16(n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        t = (torch.randn(n, config.dim, generator=g) * 0.02).to(torch.bfloat16)
        # GGUFWriter expects the raw element buffer; view bf16 as uint16 so the
        # 2-bytes-per-element layout is written verbatim (a float32 array would
        # be misread as bf16 bytes).
        return t.view(torch.int16).numpy().view(np.uint16), t

    def f32(*shape):
        return torch.ones(shape, dtype=torch.float32).numpy()

    writer = gguf.GGUFWriter(path, "muse_glimmer")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    ffn = config.ffn_dim
    seed = 0
    for i in range(config.n_layers):
        for leaf, n in (
            ("attn_q", q_dim),
            ("attn_k", kv_dim),
            ("attn_output_gate", q_dim),
            ("ffn_gate", ffn),
            ("ffn_up", ffn),
            ("attn_v", kv_dim),
            ("ffn_down", config.dim),
            ("attn_output", config.dim),
        ):
            blob, _ = bf16(n, seed)
            seed += 1
            writer.add_tensor(
                f"blk.{i}.{leaf}.weight",
                blob,
                raw_dtype=gguf.GGMLQuantizationType.BF16,
            )
        writer.add_tensor(f"blk.{i}.attn_norm.weight", f32(config.dim))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", f32(config.dim))
        writer.add_tensor(f"blk.{i}.post_attention_norm.weight", f32(config.dim))
        writer.add_tensor(f"blk.{i}.post_ffw_norm.weight", f32(config.dim))
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", f32(config.head_dim))
        writer.add_tensor(f"blk.{i}.attn_k_norm.weight", f32(config.head_dim))

    emb, _ = bf16(config.vocab_size, seed)
    seed += 1
    writer.add_tensor(
        "token_embd.weight", emb, raw_dtype=gguf.GGMLQuantizationType.BF16
    )
    out, _ = bf16(config.vocab_size, seed)
    seed += 1
    writer.add_tensor("output.weight", out, raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.add_tensor("output_norm.weight", f32(config.dim))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@unittest.skipUnless(HAS_QUANT_DEPS, "torchao quantization dependencies not available")
class TestGgufBf16Test(unittest.TestCase):
    """A MOSTLY_BF16 GGUF loads as plain bf16 (no int4 packing).

    The fused ``qko_proj`` / ``gate_up_proj`` groups arrive as plain bf16
    shards (``iter_gguf`` does not wrap bf16), so ``_flush_fused_group`` must
    row-concatenate them as bf16 rather than raising or int4-packing. Packing
    is CPU-safe; only kernel execution needs CUDA.
    """

    def setUp(self):
        try:
            import gguf  # noqa: F401
        except ImportError:
            self.skipTest("gguf package required")

    def _load(self):
        from executorch.examples.models.muse_glimmer.loaders.checkpoint_loader import (
            load_gguf_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny_bf16.gguf")
            build_muse_glimmer_gguf_bf16(path)
            return load_gguf_model(path, backend="cuda", config=GGUF_CONFIG)

    def test_all_weights_plain_bf16(self):
        """Every weight loads as a plain bf16 tensor (no torchao subclass)."""
        model, _ = self._load()
        q_dim = GGUF_CONFIG.n_heads * GGUF_CONFIG.head_dim
        kv_dim = GGUF_CONFIG.n_kv_heads * GGUF_CONFIG.head_dim
        og_dim = q_dim  # use_attn_o_gate

        # Fused groups: plain bf16, concatenated to the model's [Q|K|OG] /
        # [gate|up] row layout.
        qko = model.layers[0].self_attn.qko_proj.weight.data
        self.assertIs(type(qko), torch.Tensor)
        self.assertEqual(qko.dtype, torch.bfloat16)
        self.assertEqual(qko.shape, (q_dim + kv_dim + og_dim, GGUF_CONFIG.dim))

        gate_up = model.layers[0].mlp.gate_up_proj.weight.data
        self.assertIs(type(gate_up), torch.Tensor)
        self.assertEqual(gate_up.dtype, torch.bfloat16)
        self.assertEqual(gate_up.shape, (2 * GGUF_CONFIG.ffn_dim, GGUF_CONFIG.dim))

        # Standalone + top-level weights: plain bf16, no subclass.
        for w in (
            model.layers[0].self_attn.v_proj.weight.data,
            model.layers[0].self_attn.o_proj.weight.data,
            model.layers[0].mlp.down_proj.weight.data,
            model.lm_head.weight.data,
            model.embed_tokens.weight.data,
        ):
            self.assertIs(type(w), torch.Tensor)
            self.assertEqual(w.dtype, torch.bfloat16)

    def test_forward_runs_on_cpu(self):
        """A bf16-loaded model produces finite logits (plain nn.Linear path)."""
        model, _ = self._load()
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        input_pos = torch.arange(4, dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens, input_pos)
        self.assertEqual(logits.shape, (1, 4, GGUF_CONFIG.vocab_size))
        self.assertFalse(logits.isnan().any())
        self.assertFalse(logits.isinf().any())


if __name__ == "__main__":
    unittest.main()
