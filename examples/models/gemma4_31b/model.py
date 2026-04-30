# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 31B-IT — export-friendly reference implementation for ExecuTorch.

Model definition designed for torch.export(strict=True) with the CUDA backend.
All stateful buffers (KV cache, RoPE inv_freq) are registered buffers so they
are captured by share_mutable_buffers across prefill/decode. The numerically
sensitive primitives — RMSNorm, GELU-tanh MLP, proportional/full RoPE, and
the BHSD KV cache — are imported from ``examples.models.gemma4.text_decoder``
so the 31B and E2B/E4B paths share them.

Reference:
  - HF transformers: src/transformers/models/gemma4/modeling_gemma4.py
  - vLLM:            vllm/model_executor/models/gemma4.py

Architecture highlights for the 31B dense variant:
  - 60 decoder layers with hybrid attention: every 6th layer is "full" attention
    (idx 5, 11, ..., 59 — 10 layers); the remaining 50 use sliding-window
    attention with window=1024.
  - Sliding layers: head_dim=256, num_kv_heads=16, full RoPE, theta=10000.
  - Full layers:    head_dim=512, num_kv_heads=4, K=V (no v_proj), and
    "proportional" partial RoPE (factor=0.25, theta=1_000_000).
  - Q-norm and K-norm with learnable scale; V-norm without scale.
  - Per-layer scalar (loaded buffer) multiplied at the end of each layer.
  - Final logits are soft-capped: tanh(logits / 30) * 30.
  - Embedding is scaled by sqrt(hidden_size) before layer 0.
  - Embedding and lm_head are tied (a single weight, untied for quantization
    in the export step so lm_head can be 4-bit).
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

# Shared primitives lifted out of the gemma4 (E2B/E4B) example. These are the
# bits whose semantics are identical for both variants — RMSNorm, the GELU-tanh
# MLP, the proportional/full RoPE table builder, and the BHSD KV cache.
from executorch.examples.models.gemma4.text_decoder import (
    apply_rotary_emb,
    Gemma4KVCache,
    Gemma4MLP,
    precompute_freqs_cis,
    RMSNorm,
    RMSNormNoWeight,
)
from executorch.examples.models.gemma4_31b.sampler import sample
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Ring-buffer KV cache for sliding window attention


class RingKVCache(nn.Module):
    """Ring-buffer KV cache for sliding window attention.

    Sized to ``window_size * 2`` (not ``max_seq_len``), saving memory for
    long sequences. Positions wrap via modulo; old entries outside the
    window are masked out by ``_build_masks``.
    """

    def __init__(
        self,
        max_batch_size: int,
        window_size: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.window_size = window_size
        self.buf_size = window_size * 2
        cache_shape = (max_batch_size, num_kv_heads, self.buf_size, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape), persistent=False)

    def update(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # seq_len must not exceed buf_size, otherwise wrapped indices contain
        # duplicates and index_copy_ is non-deterministic on CUDA. The C++
        # runner must chunk prefill to respect this limit.
        assert (
            input_pos.shape[0] <= self.buf_size
        ), f"seq_len {input_pos.shape[0]} > buf_size {self.buf_size}"
        wrapped = input_pos % self.buf_size
        self.k_cache.index_copy_(2, wrapped, k_val)
        self.v_cache.index_copy_(2, wrapped, v_val)
        return self.k_cache, self.v_cache


# ---------------------------------------------------------------------------
# Config


@dataclass
class Gemma4_31BConfig:
    # Embedding / shape
    vocab_size: int = 262144
    hidden_size: int = 5376
    intermediate_size: int = 21504
    num_hidden_layers: int = 60

    # Attention shape (sliding layers — also the "default" path)
    num_attention_heads: int = 32
    num_key_value_heads: int = 16
    head_dim: int = 256

    # Attention shape (full-attention layers)
    num_global_key_value_heads: int = 4
    global_head_dim: int = 512
    attention_k_eq_v: bool = (
        True  # full layers: V is derived from the same projection as K
    )

    # RoPE — split per layer type
    sliding_rope_theta: float = 10_000.0
    full_rope_theta: float = 1_000_000.0
    full_partial_rotary_factor: float = 0.25  # proportional RoPE for full attention

    # Norm / activation
    rms_norm_eps: float = 1e-6
    hidden_activation: str = "gelu_pytorch_tanh"

    # Sampling / output
    final_logit_softcapping: float = 30.0
    tie_word_embeddings: bool = True

    # Sliding window
    sliding_window: int = 1024

    # Hybrid attention pattern
    layer_types: list = field(default_factory=list)

    # Runtime
    max_seq_len: int = 4096

    def __post_init__(self):
        if not self.layer_types:
            # Default hybrid pattern: 5 sliding then 1 full, repeated.
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length {len(self.layer_types)} != "
                f"num_hidden_layers {self.num_hidden_layers}"
            )

    @staticmethod
    def from_hf_config(config_path: str) -> "Gemma4_31BConfig":
        with open(config_path, "r") as f:
            cfg = json.load(f)
        if "text_config" in cfg:
            cfg = cfg["text_config"]

        rope_params = cfg.get("rope_parameters", {})
        sliding_rope = rope_params.get("sliding_attention", {})
        full_rope = rope_params.get("full_attention", {})

        return Gemma4_31BConfig(
            vocab_size=cfg.get("vocab_size", 262144),
            hidden_size=cfg.get("hidden_size", 5376),
            intermediate_size=cfg.get("intermediate_size", 21504),
            num_hidden_layers=cfg.get("num_hidden_layers", 60),
            num_attention_heads=cfg.get("num_attention_heads", 32),
            num_key_value_heads=cfg.get("num_key_value_heads", 16),
            head_dim=cfg.get("head_dim", 256),
            num_global_key_value_heads=cfg.get("num_global_key_value_heads", 4),
            global_head_dim=cfg.get("global_head_dim", 512),
            attention_k_eq_v=cfg.get("attention_k_eq_v", True),
            sliding_rope_theta=sliding_rope.get("rope_theta", 10_000.0),
            full_rope_theta=full_rope.get("rope_theta", 1_000_000.0),
            full_partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            hidden_activation=cfg.get("hidden_activation", "gelu_pytorch_tanh"),
            final_logit_softcapping=cfg.get("final_logit_softcapping", 30.0),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            sliding_window=cfg.get("sliding_window", 1024),
            layer_types=cfg.get("layer_types", []),
        )


# ---------------------------------------------------------------------------
# Attention — single class, branches on layer type via config
#
# RMSNorm, Gemma4MLP, the RoPE helpers, and Gemma4KVCache are imported from
# examples.models.gemma4.text_decoder so the two Gemma 4 variants share their
# numerically-sensitive primitives.


class Gemma4Attention(nn.Module):
    """Gemma 4 attention with QK-norm, per-layer head_dim, RoPE, KV cache, and SDPA.

    The same class handles both sliding and full attention; the per-layer
    config picks head_dim, num_kv_heads, RoPE flavor, and the K=V optimization.
    """

    def __init__(self, config: Gemma4_31BConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"

        if self.is_sliding:
            self.head_dim = config.head_dim
            self.n_kv_heads = config.num_key_value_heads
            self.rope_theta = config.sliding_rope_theta
            self.partial_rotary = 1.0
            self.k_eq_v = False
        else:
            self.head_dim = config.global_head_dim
            self.n_kv_heads = config.num_global_key_value_heads
            self.rope_theta = config.full_rope_theta
            self.partial_rotary = config.full_partial_rotary_factor
            self.k_eq_v = config.attention_k_eq_v

        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.scaling = 1.0  # Gemma 4 uses scale=1; QK-norm handles normalization.

        # Linear projections. v_proj is omitted on K=V layers to match the checkpoint.
        self.q_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        if not self.k_eq_v:
            self.v_proj = nn.Linear(
                self.hidden_size, self.n_kv_heads * self.head_dim, bias=False
            )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Q/K norm have learnable weight; V norm is weightless.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoWeight(self.head_dim, eps=config.rms_norm_eps)

        # Precomputed RoPE table for this layer (per-layer because head_dim
        # and theta differ between sliding and full attention). For full
        # attention layers we pass freq_base_dim=head_dim so the zero-padded
        # inv_freq matches HF's "proportional" partial RoPE.
        if self.is_sliding:
            rotary_dim = self.head_dim
            freq_base_dim = None
        else:
            rotary_dim = int(self.head_dim * self.partial_rotary)
            freq_base_dim = self.head_dim
        freqs_cos, freqs_sin = precompute_freqs_cis(
            rotary_dim,
            config.max_seq_len,
            theta=self.rope_theta,
            freq_base_dim=freq_base_dim,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # KV cache. Sliding layers use a ring buffer (2x window) to save
        # memory; full layers use a flat buffer (max_seq_len).
        if self.is_sliding:
            self.kv_cache = RingKVCache(
                max_batch_size=1,
                window_size=config.sliding_window,
                num_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            self.kv_cache = Gemma4KVCache(
                max_batch_size=1,
                max_seq_len=config.max_seq_len,
                num_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                use_index_copy=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        # raw_kv is the linear output before any norm — needed for K=V layers
        # so V can be derived from the same tensor as K (post-norm differently).
        raw_k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        if self.k_eq_v:
            raw_v = raw_k
        else:
            raw_v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Norms applied per-head (HF unflatten -> norm -> flatten pattern).
        q = self.q_norm(q)
        k = self.k_norm(raw_k)
        v = self.v_norm(raw_v)

        # Move to BHSD for SDPA / KV cache.
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE on Q and K only (V is not rotated). cos/sin are gathered for
        # the current positions to avoid baking the full table into the graph.
        cos = self.freqs_cos[input_pos]
        sin = self.freqs_sin[input_pos]
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Update cache and read back full K/V.
        k, v = self.kv_cache.update(input_pos, k, v)

        # SDPA with explicit additive mask (already includes causal +
        # sliding-window masking; built once per forward at the model level).
        # `scale=1.0` matches HF Gemma 4 — Q-norm/K-norm have absorbed the
        # 1/sqrt(d) factor into their trained weights, so the standard SDPA
        # default of 1/sqrt(head_dim) would over-divide. enable_gqa lets the
        # kernel handle the head ratio without us materializing expanded K/V.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=True,
            scale=self.scaling,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)


# ---------------------------------------------------------------------------
# Decoder block — Gemma's "norm sandwich" pattern.


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4_31BConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config.hidden_size, config.intermediate_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Per-layer scalar (loaded from checkpoint) — multiplied at the end of
        # each layer. Kept as a buffer (not nn.Parameter) so it isn't quantized.
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        sliding_mask: torch.Tensor,
        full_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = sliding_mask if self.is_sliding else full_mask

        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, input_pos, attn_mask)
        h = self.post_attention_layernorm(h)
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h

        return x * self.layer_scalar


# ---------------------------------------------------------------------------
# Top-level model


class Gemma4_31B(nn.Module):
    def __init__(self, config: Gemma4_31BConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Held separately so it can be untied + quantized at export time.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Constants (registered as buffers so they move with .to(device)).
        self.register_buffer(
            "embed_normalizer",
            torch.tensor(config.hidden_size**0.5),
            persistent=False,
        )
        self.register_buffer(
            "logit_softcap",
            torch.tensor(config.final_logit_softcapping),
            persistent=False,
        )
        # cache_positions[i] = i — used to build attention masks without
        # introducing dynamic-shape tensors at runtime.
        self.register_buffer(
            "cache_positions",
            torch.arange(config.max_seq_len, dtype=torch.long),
            persistent=False,
        )

    def _build_masks(
        self, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build boolean (B=1, H=1, T_q, T_kv) masks for full and sliding attention.

        True = attend.  Built once per forward, shared across layers of the
        same type.  Full mask is (T_q, max_seq_len); sliding mask is
        (T_q, buf_size) where buf_size = 2 * sliding_window.
        """
        # Full attention mask: (T_q, max_seq_len)
        cache_pos = self.cache_positions  # (max_seq_len,)
        q_pos = input_pos.unsqueeze(1)  # (T_q, 1)
        causal = q_pos >= cache_pos.unsqueeze(0)
        full_mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, max_seq_len)

        # Sliding attention mask over ring buffer: (T_q, buf_size)
        buf_size = self.config.sliding_window * 2
        seq_len = input_pos.shape[0]
        total_written = input_pos[0] + seq_len
        j = torch.arange(buf_size, dtype=torch.long, device=input_pos.device)
        ring_pos = j + ((total_written - 1 - j) // buf_size) * buf_size
        delta = q_pos - ring_pos.unsqueeze(0)
        sliding = (ring_pos >= 0) & (delta >= 0) & (delta < self.config.sliding_window)
        sliding_mask = sliding.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, buf_size)

        return sliding_mask, full_mask

    def forward(
        self,
        tokens: torch.LongTensor,
        input_pos: torch.LongTensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the model.

        Args:
            tokens: (B, T) token IDs.
            input_pos: (T,) absolute positions for RoPE / KV cache.
            temperature: optional 1-D float tensor controlling on-device sampling.
                When provided, returns sampled tokens (B, 1) via Gumbel-max;
                when None (e.g. eager eval), returns full logits (B, T, V) with
                soft-capping applied so callers see post-cap values.

        Returns:
            (B, 1) token IDs when sampling, else (B, T, V) float32 logits.
        """
        x = self.embed_tokens(tokens) * self.embed_normalizer

        sliding_mask, full_mask = self._build_masks(input_pos)
        for layer in self.layers:
            x = layer(x, input_pos, sliding_mask, full_mask)

        x = self.norm(x)

        if temperature is None:
            logits = self.lm_head(x).float()
            cap = self.logit_softcap.float()
            return torch.tanh(logits / cap) * cap

        # Decode-time fast path: only materialize logits for the last token.
        last = self.lm_head(x[:, -1, :]).float()
        cap = self.logit_softcap.float()
        last = torch.tanh(last / cap) * cap
        return sample(last, temperature)

    # ---------------- checkpoint loading ----------------

    @staticmethod
    def from_hf_checkpoint(
        model_dir: str, max_seq_len: int = 4096
    ) -> tuple["Gemma4_31B", Gemma4_31BConfig]:
        """Build the model on `meta` and load weights from the HF safetensors checkpoint.

        Uses lazy shard-by-shard loading + assign=True so peak memory stays at
        roughly one shard's worth of weights.
        """
        config = Gemma4_31BConfig.from_hf_config(os.path.join(model_dir, "config.json"))
        config.max_seq_len = max_seq_len

        print(
            f"Building Gemma4_31B on meta (layers={config.num_hidden_layers}, "
            f"hidden={config.hidden_size}, max_seq_len={max_seq_len})..."
        )
        with torch.device("meta"):
            model = Gemma4_31B(config)

        print(f"Loading weights from {model_dir}...")
        state_dict = _load_and_remap_checkpoint(model_dir, config)

        # Tied embeddings: copy embedding weight into lm_head when missing.
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )

        # Runtime buffers (KV caches, RoPE tables, masks) are zero-initialized
        # and not in the checkpoint — those are the "expected" missing keys.
        runtime_prefixes = (
            ".kv_cache.",
            ".freqs_cos",
            ".freqs_sin",
            "embed_normalizer",
            "logit_softcap",
            "cache_positions",
        )
        actual_missing = set(missing)
        expected = {k for k in actual_missing if any(p in k for p in runtime_prefixes)}
        extra = actual_missing - expected
        if extra:
            print(f"  WARNING: missing weight keys: {sorted(extra)[:10]}")
        if unexpected:
            print(f"  WARNING: unexpected keys: {sorted(unexpected)[:10]}")
        print(
            f"  Loaded {len(state_dict)} tensors "
            f"({len(expected)} runtime buffers OK)"
        )
        return model, config


# ---------------------------------------------------------------------------
# Weight loading utilities


# HuggingFace key -> our model key.  Patterns use `{}` for the layer index.
_HF_KEY_MAP = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
    # Per-layer norms
    "model.layers.{}.input_layernorm.weight": "layers.{}.input_layernorm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
    "model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.pre_feedforward_layernorm.weight",
    "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_feedforward_layernorm.weight",
    "model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
    # Attention projections
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.self_attn.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.self_attn.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.self_attn.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.self_attn.o_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.self_attn.q_norm.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.self_attn.k_norm.weight",
    # MLP
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.gate_proj.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.up_proj.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.down_proj.weight",
}

# Multimodal keys we deliberately ignore for the text-only export.
_IGNORED_PREFIXES = (
    "model.vision_tower.",
    "model.embed_vision.",
)


def _hf_to_model_key(hf_key: str) -> Optional[str]:
    # Gemma4ForConditionalGeneration stores the LM under model.language_model.*
    norm = hf_key
    if norm.startswith("model.language_model."):
        norm = norm.replace("model.language_model.", "model.", 1)

    if norm.startswith(_IGNORED_PREFIXES):
        return None

    for hf_pat, model_pat in _HF_KEY_MAP.items():
        if "{}" not in hf_pat:
            if norm == hf_pat:
                return model_pat
            continue
        regex = re.escape(hf_pat).replace(r"\{\}", r"(\d+)")
        m = re.fullmatch(regex, norm)
        if m:
            return model_pat.replace("{}", m.group(1), 1)
    return None


def _load_and_remap_checkpoint(model_dir: str, config: Gemma4_31BConfig) -> dict:
    """Stream-load safetensors shards and remap keys to model state_dict keys."""
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    elif os.path.exists(os.path.join(model_dir, "model.safetensors")):
        shard_files = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors checkpoint in {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    skipped = 0
    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for ckpt_key in f.keys():
                model_key = _hf_to_model_key(ckpt_key)
                if model_key is None:
                    skipped += 1
                    continue
                tensor = f.get_tensor(ckpt_key)
                # layer_scalar in checkpoint is shape (1,) bf16 — keep as-is.
                state_dict[model_key] = tensor
    if skipped > 0:
        print(f"  Skipped {skipped} non-text keys (vision tower, etc.)")
    return state_dict


# ---------------------------------------------------------------------------
# Runtime buffer materialization


def materialize_runtime_buffers(
    model: Gemma4_31B,
    dtype: torch.dtype,
    device: str = "cpu",
) -> None:
    """Replace meta-device buffers with real tensors and set runtime constants.

    Called after weight loading to fill in KV caches (zeros), RoPE tables
    (computed), and scalar constants. Only touches buffers still on the meta
    device — loaded (non-meta) buffers are left in place.
    """
    config = model.config

    for fqn, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        parts = fqn.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        is_kv = ".kv_cache." in fqn
        target_dtype = dtype if is_kv else torch.float32
        if buf.dtype == torch.bool:
            target_dtype = torch.bool
        parent.register_buffer(
            parts[-1],
            torch.zeros(buf.shape, dtype=target_dtype, device=device),
            persistent=False,
        )

    for layer in model.layers:
        attn = layer.self_attn
        if attn.is_sliding:
            rotary_dim, freq_base_dim = attn.head_dim, None
        else:
            rotary_dim = int(attn.head_dim * attn.partial_rotary)
            freq_base_dim = attn.head_dim
        cos, sin = precompute_freqs_cis(
            rotary_dim,
            config.max_seq_len,
            theta=attn.rope_theta,
            freq_base_dim=freq_base_dim,
        )
        attn.register_buffer("freqs_cos", cos.to(device), persistent=False)
        attn.register_buffer("freqs_sin", sin.to(device), persistent=False)

    model.register_buffer(
        "embed_normalizer",
        torch.tensor(config.hidden_size**0.5, device=device),
        persistent=False,
    )
    model.register_buffer(
        "logit_softcap",
        torch.tensor(config.final_logit_softcapping, device=device),
        persistent=False,
    )
    model.register_buffer(
        "cache_positions",
        torch.arange(config.max_seq_len, dtype=torch.long, device=device),
        persistent=False,
    )
