"""
Export-friendly Gemma4 text model (self-contained, no HuggingFace deps).

Mirrors HuggingFace `Gemma4TextModel` (modeling_gemma4.py main branch) for the
text-only path of Gemma4-31B. Designed for torch.export(strict=True) on CUDA:
all stateful tensors (KV caches, RoPE tables, causal masks) are registered
buffers; in-place cache updates are wired so prefill/decode methods can share
the same buffers via share_mutable_buffers=True.

Two attention configurations live side-by-side per decoder layer:
  - sliding_attention (50/60 layers): head_dim=256, 16 KV heads, separate
    Q/K/V projections, full RoPE (theta=10k), sliding causal mask (window=1024).
  - full_attention   (10/60 layers): head_dim=512, 4 KV heads, V reuses K
    (attention_k_eq_v), partial RoPE (25%, theta=1M), full causal mask.
"""

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config


@dataclass
class Gemma4TextConfig:
    vocab_size: int = 262144
    hidden_size: int = 5376
    intermediate_size: int = 21504
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    num_key_value_heads: int = 16
    num_global_key_value_heads: int = 4
    head_dim: int = 256
    global_head_dim: int = 512
    sliding_window: int = 1024
    rms_norm_eps: float = 1e-6
    sliding_rope_theta: float = 10_000.0
    full_rope_theta: float = 1_000_000.0
    full_partial_rotary_factor: float = 0.25
    attention_k_eq_v: bool = True
    final_logit_softcapping: float = 30.0
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    max_seq_len: int = 4096
    layer_types: list = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 6 != 0 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    @staticmethod
    def from_hf_config(config_path: str) -> "Gemma4TextConfig":
        with open(config_path, "r") as f:
            cfg = json.load(f)
        if "text_config" in cfg:
            cfg = cfg["text_config"]
        rope_params = cfg.get("rope_parameters", {})
        sliding_rope = rope_params.get("sliding_attention", {})
        full_rope = rope_params.get("full_attention", {})
        return Gemma4TextConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            num_global_key_value_heads=cfg.get("num_global_key_value_heads", 4),
            head_dim=cfg["head_dim"],
            global_head_dim=cfg.get("global_head_dim", cfg["head_dim"]),
            sliding_window=cfg["sliding_window"],
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            sliding_rope_theta=sliding_rope.get("rope_theta", 10_000.0),
            full_rope_theta=full_rope.get("rope_theta", 1_000_000.0),
            full_partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
            attention_k_eq_v=cfg.get("attention_k_eq_v", True),
            final_logit_softcapping=cfg.get("final_logit_softcapping", 30.0),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            pad_token_id=cfg.get("pad_token_id", 0),
            layer_types=cfg.get("layer_types"),
        )


# ---------------------------------------------------------------------------
# Normalization (Gemma4: standard RMSNorm, NOT the unit-offset Gemma2/3 variant)


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            rms = rms * self.weight.float()
        return rms.to(x.dtype)


# ---------------------------------------------------------------------------
# Scaled embedding


class Gemma4ScaledEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.padding_idx = padding_idx
        self.embed_scale = float(hidden_size) ** 0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = F.embedding(input_ids, self.weight, padding_idx=self.padding_idx)
        return out * self.embed_scale


# ---------------------------------------------------------------------------
# RoPE (full and partial variants share the same apply_rotary helper)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, D_rot); cos/sin: (T, D_rot)."""
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D_rot)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# KV Cache — wrapped in its own nn.Module so torch.export sees a clean
# method-call boundary for the in-place update + read. Mirrors the qwen3_5_moe
# KVCache pattern. Wrapping (vs. registering buffers directly on attention)
# is what lets AOTI expose these buffers as user-managed constants with
# stable FQNs, which the CUDA backend's load_constants_with_cache() then
# shares across the prefill and decode methods at runtime.


class KVCache(nn.Module):
    def __init__(self, num_kv_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        self.register_buffer(
            "k_cache",
            torch.zeros(1, num_kv_heads, max_seq_len, head_dim),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(1, num_kv_heads, max_seq_len, head_dim),
        )

    def update(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.k_cache[:, :, input_pos] = k
        self.v_cache[:, :, input_pos] = v
        return self.k_cache, self.v_cache


# ---------------------------------------------------------------------------
# Attention


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.use_alternative_attention = (
            config.attention_k_eq_v and not self.is_sliding
        )

        self.head_dim = (
            config.head_dim if self.is_sliding else config.global_head_dim
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = (
            config.num_key_value_heads
            if self.is_sliding
            else config.num_global_key_value_heads
        )
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        # Gemma4 uses scaling=1.0 (NOT head_dim**-0.5). The Q/K magnitudes
        # are normalized by q_norm / k_norm whose learned weights implicitly
        # encode the right pre-softmax scale; adding 1/sqrt(d) on top
        # double-scales and collapses softmax to near-uniform, producing
        # repetitive single-token output.
        self.scaling = 1.0
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # Partial RoPE: only the first `rotary_dim` channels of each head get rotated.
        if self.is_sliding:
            self.rotary_dim = self.head_dim  # full RoPE
            rope_theta = config.sliding_rope_theta
        else:
            self.rotary_dim = int(self.head_dim * config.full_partial_rotary_factor)
            rope_theta = config.full_rope_theta

        # Q always has its own projection.
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        # For full attention with attention_k_eq_v, V reuses K's projection output.
        if self.use_alternative_attention:
            self.v_proj = None
        else:
            self.v_proj = nn.Linear(
                config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
            )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=True)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=True)
        # v_norm has NO learnable weight in Gemma4.
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        # KV cache as a submodule (qwen pattern) so AOTI gets a clean
        # update() boundary; needed for cross-method buffer sharing.
        self.kv_cache = KVCache(
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
        )

        # RoPE inv_freq (precomputed). Cos/sin tables are derived in forward
        # so they pick up the runtime input_pos.
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

        # Causal / sliding mask, indexed by query position at runtime.
        # mask[i, j] = True (allow) iff j <= i AND (sliding => i - j < sliding_window).
        positions = torch.arange(config.max_seq_len)
        causal = positions[None, :] <= positions[:, None]  # (L, L)
        if self.is_sliding:
            within_window = (positions[:, None] - positions[None, :]) < config.sliding_window
            mask = causal & within_window
        else:
            mask = causal
        self.register_buffer("attn_mask", mask)

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to Q, K, V. For k_eq_v, V reuses the K projection output
        # (cloned to avoid tensor-aliasing pitfalls under torch.export).
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        if self.v_proj is not None:
            v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        else:
            v = k.clone()

        # Normalize Q/K (with learnable scale), V (no learnable scale).
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Rotary embedding (partial for full layers).
        positions_f = input_pos.to(torch.float32)
        freqs = torch.outer(positions_f, self.inv_freq)  # (T, rotary_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, rotary_dim)
        cos = emb.cos().to(q.dtype)
        sin = emb.sin().to(q.dtype)
        if self.rotary_dim == self.head_dim:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        else:
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q = torch.cat([_apply_rope(q_rot, cos, sin), q_pass], dim=-1)
            k = torch.cat([_apply_rope(k_rot, cos, sin), k_pass], dim=-1)

        # Transpose to (B, H, T, D) for attention and cache update.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Write current K/V into cache via submodule update method, which
        # returns the (full) cache tensors. Using a method-call boundary
        # (vs. inline setitem + separate self.k_cache reads) is what gets
        # AOTI to expose k_cache/v_cache as user-managed constants with
        # stable FQNs — the prerequisite for cross-method sharing.
        k_full, v_full = self.kv_cache.update(input_pos, k, v)

        # Build (T, max_seq_len) mask, then broadcast to attn shape.
        attn_mask = self.attn_mask[input_pos].unsqueeze(0).unsqueeze(0)

        # GQA: SDPA's enable_gqa handles num_kv_heads != num_heads.
        out = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=attn_mask,
            scale=self.scaling,
            enable_gqa=True,
        )
        # (B, H, T, D) -> (B, T, H*D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MLP (SwiGLU with tanh-approx GELU)


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder block


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, input_pos)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x * self.layer_scalar


# ---------------------------------------------------------------------------
# Top-level model


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Gemma4ScaledEmbedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # lm_head is tied to embed_tokens; we expose a Linear and tie weights below.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.final_logit_softcapping = config.final_logit_softcapping

    def forward(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        h = self.embed_tokens(tokens)
        for layer in self.layers:
            h = layer(h, input_pos)
        h = self.norm(h)
        logits = self.lm_head(h)
        if self.final_logit_softcapping:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    @staticmethod
    def from_hf_checkpoint(model_dir: str, max_seq_len: int = 4096):
        """Build model on meta device, then load+remap HF safetensors weights."""
        from executorch.examples.models.gemma4.convert_weights import (
            load_and_remap_checkpoint,
        )

        config_path = os.path.join(model_dir, "config.json")
        config = Gemma4TextConfig.from_hf_config(config_path)
        config.max_seq_len = max_seq_len

        with torch.device("meta"):
            model = Gemma4TextModel(config)

        state_dict = load_and_remap_checkpoint(model_dir, config)

        # tie_word_embeddings: HF checkpoint may omit lm_head.weight.
        if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        # Buffers (KV caches, masks, inv_freq) and v_proj for full layers
        # are not in the checkpoint — they're materialized later.
        runtime_prefixes = (
            ".k_cache",
            ".v_cache",
            ".attn_mask",
            ".inv_freq",
            ".layer_scalar",
        )
        weight_missing = [
            k for k in missing if not any(p in k for p in runtime_prefixes)
        ]
        # Full attention layers legitimately have no v_proj weight.
        weight_missing = [k for k in weight_missing if ".v_proj." not in k]
        if weight_missing:
            print(f"WARNING: missing weights: {weight_missing[:8]}{'...' if len(weight_missing) > 8 else ''}")
        if unexpected:
            print(f"WARNING: unexpected keys: {sorted(unexpected)[:8]}")

        return model, config
