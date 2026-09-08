# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-independent Muse Glimmer text model for ExecuTorch export.

CUDA and MLX adaptations are applied by their source-transformation modules.
"""

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


# Config


@dataclass
class MuseGlimmerConfig:
    dim: int = 6656
    n_layers: int = 52
    n_heads: int = 32
    n_kv_heads: int = 2
    head_dim: int = 128
    ffn_dim_multiplier: float = 3.0
    vocab_size: int = 202048
    norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    use_qk_norm: bool = True
    use_attn_o_gate: bool = True
    fuse_qkv: bool = True
    # CUDA GGUF layout: fuse Q/K/output-gate into one projection but keep V
    # separate so it can carry its own per-layer quant bit-width. Takes
    # precedence over ``fuse_qkv`` when True.
    fuse_qko: bool = False
    fuse_gate_up: bool = True
    output_soft_cap_temp: float = 20.0
    global_attn_cfg: str = "[2048,2048,2048,0]"
    every_n_layers_nope: int = 4
    normalize_tok_embeddings: bool = True
    output_multiplier: float = 0.19611613513
    post_norm_eps: float = 1e-8
    max_seq_len: int = 16384

    @property
    def ffn_dim(self) -> int:
        return int(self.ffn_dim_multiplier * self.dim)

    @property
    def attn_pattern(self) -> list[int]:
        s = self.global_attn_cfg.strip("[]")
        return [int(x) for x in s.split(",")]

    def layer_use_rope(self, layer_idx: int) -> bool:
        return (self.n_layers - layer_idx - 1) % self.every_n_layers_nope != 0

    def layer_window_size(self, layer_idx: int) -> int:
        """Return window size for layer (0 = global attention)."""
        nope_freq = self.every_n_layers_nope
        count_backward = layer_idx + nope_freq - self.n_layers % nope_freq
        pattern = self.attn_pattern
        return pattern[count_backward % len(pattern)]

    @staticmethod
    def from_json(path: str) -> "MuseGlimmerConfig":
        with open(path) as f:
            d = json.load(f)
        return MuseGlimmerConfig(
            **{
                k: v
                for k, v in d.items()
                if k in MuseGlimmerConfig.__dataclass_fields__
            }
        )


# Norms
#
# Norms are expressed canonically with ``F.rms_norm`` (a single op). The MLX
# backend consumes this directly (fused via ``FuseRMSNormPass``); the CUDA
# backend rewrites it to the decomposed elementwise form in
# ``cuda_source_transformations``.


class RMSNorm(nn.Module):
    """RMSNorm with a learnable scale, expressed via ``F.rms_norm``."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class RMSNormNoWeight(nn.Module):
    """Scaleless RMSNorm (QK-norm, embedding norm), expressed via ``F.rms_norm``."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), None, self.eps)


# Rotary Position Embeddings (interleaved cos/sin — no complex tensors)


# Dtypes whose mantissa preserves RoPE phase well enough to rotate in place,
# used only when a caller opts in via ``allow_low_precision`` (see
# rotate_interleaved). fp16 has a 10-bit mantissa (~1e-3); bf16 is deliberately
# absent because its 8-bit mantissa quantizes cos/sin too coarsely.
ROPE_SAFE_DTYPES: frozenset[torch.dtype] = frozenset({torch.float32, torch.float16})


def rotate_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    allow_low_precision: bool = False,
) -> torch.Tensor:
    """Apply interleaved (adjacent-pair) rotary embeddings. x: [B, H, T, D].

    Pairs dims (2i, 2i+1) — matches ggml GGML_ROPE_TYPE_NORMAL / MLX
    traditional=True. cos, sin: [..., T, D//2].

    By default the rotation runs in fp32 (cast in, cast back) regardless of x's
    dtype, matching the eager complex reference. A caller whose positions are
    small and whose cos/sin are precomputed in fp32 -- the vision tower, on a
    32x32 grid -- may pass ``allow_low_precision=True`` to rotate natively when
    x is in ``ROPE_SAFE_DTYPES`` (fp16/fp32), which drops the surrounding casts.
    The decoder does not opt in: its long context makes the fp16 phase error
    unverified, so it stays on fp32.
    """
    if allow_low_precision and x.dtype in ROPE_SAFE_DTYPES:
        compute_dtype = x.dtype
    else:
        compute_dtype = torch.float32
    xc = x.to(compute_dtype)
    cos = cos.to(compute_dtype)
    sin = sin.to(compute_dtype)
    x1 = xc[..., 0::2]
    x2 = xc[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2).to(x.dtype)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply interleaved rotary embeddings to a q/k pair. q, k: [B, H, T, D]."""
    return rotate_interleaved(q, cos, sin), rotate_interleaved(k, cos, sin)


# KV Caches


class RingKVCache(nn.Module):
    """Ring-buffer KV cache for sliding window attention.

    Sized to ``window_size * 2`` to avoid index_copy_ collisions when
    prefill chunks approach the window size. The C++ runner must chunk
    prefill to not exceed buf_size.
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
        wrapped = input_pos % self.buf_size
        self.k_cache.index_copy_(2, wrapped, k_val)
        self.v_cache.index_copy_(2, wrapped, v_val)
        return self.k_cache, self.v_cache


class FlatKVCache(nn.Module):
    """Flat KV cache for global (non-sliding) attention layers."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        cache_shape = (max_batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape), persistent=False)

    def update(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.k_cache.index_copy_(2, input_pos, k_val)
        self.v_cache.index_copy_(2, input_pos, v_val)
        return self.k_cache, self.v_cache


# Attention


class MuseGlimmerAttention(nn.Module):
    """Muse Glimmer attention with fused QKV+OG, scaleless QK-norm, iRoPE, and GQA.

    Each layer is either sliding-window+RoPE or global+NoPE, determined at
    init by the config's iRoPE and attention pattern. The same class handles
    both; torch.export traces a fixed path per layer.
    """

    def __init__(self, config: MuseGlimmerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim

        self.use_rope = config.layer_use_rope(layer_idx)
        window = config.layer_window_size(layer_idx)
        self.is_sliding = window > 0
        self.window_size = window

        self.use_o_gate = config.use_attn_o_gate
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        og_dim = q_dim if self.use_o_gate else 0
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.fuse_qkv = config.fuse_qkv
        self.fuse_qko = config.fuse_qko

        # Attention input pre-norm, shared by the projections below.
        self.qkv_proj_norm = RMSNorm(config.dim, config.norm_eps)
        # Three projection layouts (all mathematically equivalent):
        #  * fuse_qko (CUDA GGUF): Q/K/output-gate fused into ``qko_proj`` (all
        #    Q4_K) plus a standalone ``v_proj`` so V can carry its own per-layer
        #    quant bit-width (the GGUF stores attn_v at Q6_K on some layers,
        #    Q4_K on others, which a single fused qkv weight cannot express).
        #  * fuse_qkv (default): one fused ``qkv_proj`` (Q+K+V+OG), one matmul.
        #  * else: separate q/k/v/og projections (quantized independently).
        if self.fuse_qko:
            self.qko_proj = nn.Linear(config.dim, q_dim + kv_dim + og_dim, bias=False)
            self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)
        elif self.fuse_qkv:
            self.qkv_proj = nn.Linear(
                config.dim, q_dim + 2 * kv_dim + og_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(config.dim, q_dim, bias=False)
            self.k_proj = nn.Linear(config.dim, kv_dim, bias=False)
            self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)
            if self.use_o_gate:
                self.og_proj = nn.Linear(config.dim, og_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.dim, bias=False)

        # Attention scale passed to SDPA. Absorbs both the QK-norm query
        # scaling (43.78.../sqrt(head_dim)) and SDPA's default 1/sqrt(head_dim),
        # avoiding a separate tensor multiply that creates a convert_element_type
        # op during AOTI constant folding.
        self.attn_scale = (43.7840518911 / math.sqrt(config.head_dim)) / math.sqrt(
            self.head_dim
        )
        if config.use_qk_norm:
            self.q_norm = RMSNormNoWeight(self.head_dim, config.norm_eps)
            self.k_norm = RMSNormNoWeight(self.head_dim, config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # RoPE inv_freq (only for RoPE layers)
        self.rope_theta = config.rope_theta
        if self.use_rope:
            self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

        # KV cache: ring buffer for SWA, flat for global
        if self.is_sliding:
            self.kv_cache = RingKVCache(
                max_batch_size=1,
                window_size=self.window_size,
                num_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            self.kv_cache = FlatKVCache(
                max_batch_size=1,
                max_seq_len=config.max_seq_len,
                num_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
            )

    def _compute_inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        """Compute RoPE inverse-frequency table for this layer."""
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Pre-norm shared by all projection layouts.
        h = self.qkv_proj_norm(x)
        if self.fuse_qko:
            # Fused Q/K/output-gate projection, plus a separate V projection.
            # Slice with [..., start:end] (decomposes to aten::slice, not
            # aten::split_copy which lacks an AOTI fallback). qko: [Q | K | OG].
            qko = self.qko_proj(h)
            q_end = self.q_dim
            k_end = q_end + self.kv_dim
            xq = qko[..., :q_end]
            xk = qko[..., q_end:k_end]
            og = qko[..., k_end:] if self.use_o_gate else None
            xv = self.v_proj(h)
        elif self.fuse_qkv:
            qkv = self.qkv_proj(h)
            # Slice Q, K, V, and optionally output gate from the fused
            # projection. Uses [..., start:end] slicing (decomposes to
            # aten::slice, not aten::split_copy which lacks an AOTI fallback).
            q_end = self.q_dim
            k_end = q_end + self.kv_dim
            v_end = k_end + self.kv_dim
            xq = qkv[..., :q_end]
            xk = qkv[..., q_end:k_end]
            xv = qkv[..., k_end:v_end]
            og = qkv[..., v_end:] if self.use_o_gate else None
        else:
            xq = self.q_proj(h)
            xk = self.k_proj(h)
            xv = self.v_proj(h)
            og = self.og_proj(h) if self.use_o_gate else None

        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim)

        # QK-norm (scaleless, no query scaling — absorbed into SDPA scale)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # Transpose to [B, H, T, D] for SDPA
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # RoPE (interleaved cos/sin)
        if self.use_rope:
            freqs = torch.outer(input_pos.float(), self.inv_freq)
            cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
            sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
            xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # Update KV cache and read back full cache
        k, v = self.kv_cache.update(input_pos, xk, xv)

        # SDPA with GQA. attn_scale absorbs both the muP query scaling
        # and the standard 1/sqrt(head_dim), matching the standalone's
        # (norm(Q) * scale_query_by) @ K^T / sqrt(head_dim).
        y = F.scaled_dot_product_attention(
            xq,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=True,
            scale=self.attn_scale,
        )
        y = y.transpose(1, 2).contiguous()

        # Output gate: sigmoid(OG) * attn_output
        if self.use_o_gate and og is not None:
            og = og.view(B, T, self.n_heads, self.head_dim)
            y = torch.sigmoid(og) * y

        y = y.reshape(B, T, -1)
        return self.o_proj(y)


# MLP (SwiGLU)


class MuseGlimmerMLP(nn.Module):
    """SwiGLU MLP with pre-norm. Gate+up can be fused or separate."""

    def __init__(self, config: MuseGlimmerConfig):
        super().__init__()
        self.intermediate_size = config.ffn_dim
        self.fuse_gate_up = config.fuse_gate_up
        self.norm = RMSNorm(config.dim, config.norm_eps)
        if self.fuse_gate_up:
            self.gate_up_proj = nn.Linear(config.dim, 2 * config.ffn_dim, bias=False)
        else:
            self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
            self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.fuse_gate_up:
            h = self.gate_up_proj(h)
            gate = h[..., : self.intermediate_size]
            up = h[..., self.intermediate_size :]
        else:
            gate = self.gate_proj(h)
            up = self.up_proj(h)
        return self.down_proj(F.silu(gate) * up)


# Decoder block


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MuseGlimmerAttention(config, layer_idx)
        self.mlp = MuseGlimmerMLP(config)
        self.post_attn_norm = RMSNorm(config.dim, config.post_norm_eps)
        self.post_ffn_norm = RMSNorm(config.dim, config.post_norm_eps)
        self.is_sliding = self.self_attn.is_sliding

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        sliding_mask: torch.Tensor,
        global_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = sliding_mask if self.is_sliding else global_mask

        attn_out = self.self_attn(x, input_pos, attn_mask)
        x = x + self.post_attn_norm(attn_out)

        ffn_out = self.mlp(x)
        x = x + self.post_ffn_norm(ffn_out)

        return x


# Top-level model


class MuseGlimmerModel(nn.Module):
    def __init__(self, config: MuseGlimmerConfig):
        super().__init__()
        self.config = config
        self.normalize_tok_embeddings = config.normalize_tok_embeddings

        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        if config.normalize_tok_embeddings:
            self.tok_norm = RMSNormNoWeight(config.dim, config.norm_eps)

        self.layers = nn.ModuleList(
            [MuseGlimmerDecoderLayer(config, i) for i in range(config.n_layers)]
        )

        self.output_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Sliding window size for mask building (from config pattern)
        self._sliding_window = max(config.attn_pattern)

        # Constants as buffers so they move with .to(device)
        self.register_buffer(
            "output_multiplier",
            torch.tensor(config.output_multiplier),
            persistent=False,
        )
        self.register_buffer(
            "logit_softcap",
            torch.tensor(config.output_soft_cap_temp),
            persistent=False,
        )
        self.register_buffer(
            "cache_positions",
            torch.arange(config.max_seq_len, dtype=torch.long),
            persistent=False,
        )

    def _build_masks(
        self, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build boolean (1, 1, T_q, T_kv) masks for global and sliding attention.

        True = attend. Built once per forward, shared across layers of the
        same type.
        """
        # Global (causal) mask: (1, 1, T_q, max_seq_len)
        cache_pos = self.cache_positions
        q_pos = input_pos.unsqueeze(1)
        causal = q_pos >= cache_pos.unsqueeze(0)
        global_mask = causal.unsqueeze(0).unsqueeze(0)

        # Sliding window mask over ring buffer: (1, 1, T_q, buf_size)
        window_size = self._sliding_window
        buf_size = window_size * 2
        seq_len = input_pos.shape[0]
        total_written = input_pos[0] + seq_len
        j = torch.arange(buf_size, dtype=torch.long, device=input_pos.device)
        ring_pos = j + ((total_written - 1 - j) // buf_size) * buf_size
        delta = q_pos - ring_pos.unsqueeze(0)
        sliding = (ring_pos >= 0) & (delta >= 0) & (delta < window_size)
        sliding_mask = sliding.unsqueeze(0).unsqueeze(0)

        return sliding_mask, global_mask

    def _soft_cap(self, logits: torch.Tensor) -> torch.Tensor:
        """Output soft-capping: cap * tanh(logits * multiplier / cap)."""
        mult = self.output_multiplier.float()
        cap = self.logit_softcap.float()
        return torch.tanh(logits.float() * mult / cap) * cap

    def embed_text(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Token embedding lookup + (optional) scaleless norm, cast to bf16.

        This is the exported ``embed_text`` method and the first stage of the
        multimodal prefill: the runner calls it to get ``inputs_embeds``, then
        (for image input) overwrites the rows at ``patch_token_id`` positions
        with vision features before calling ``prefill_from_embeds``. Numerically
        identical to the embedding stage of the original text-only ``forward``.

        Args:
            tokens: (B, T) token IDs.

        Returns:
            (B, T, dim) bfloat16 embeddings.
        """
        x = self.embed_tokens(tokens)
        if self.normalize_tok_embeddings:
            x = self.tok_norm(x)
        return x.to(torch.bfloat16)

    def _run_blocks(
        self,
        x: torch.Tensor,
        input_pos: torch.LongTensor,
    ) -> torch.Tensor:
        """Run the 52 decoder layers over pre-embedded (bf16) hidden states.

        ``x`` is the layer-0 input (already ``embed_text``-scaled bf16). Shared
        by ``prefill_from_embeds`` and ``decode_forward`` so all entry points
        run the identical decoder stack.
        """
        sliding_mask, global_mask = self._build_masks(input_pos)
        for layer in self.layers:
            x = layer(x, input_pos, sliding_mask, global_mask)
        return x

    def prefill_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_pos: torch.LongTensor,
    ) -> torch.Tensor:
        """Run the model on PRE-COMPUTED embeddings (unified prefill).

        This is the exported ``prefill`` entry: the runner builds
        ``inputs_embeds`` via ``embed_text`` (+ image splice for multimodal) and
        passes it here. Used for BOTH text-only and image+text.

        Args:
            inputs_embeds: (B, T, dim) bfloat16 embeddings.
            input_pos: (T,) absolute positions for RoPE / KV cache.

        Returns:
            (B, T, V) float32 soft-capped logits.
        """
        x = self._run_blocks(inputs_embeds, input_pos)
        x = self.output_norm(x)
        return self._soft_cap(self.lm_head(x))

    def forward(
        self,
        tokens: torch.LongTensor,
        input_pos: torch.LongTensor,
    ) -> torch.Tensor:
        """Run the model on tokens (text-only / NLL path).

        Equivalent to ``prefill_from_embeds(embed_text(tokens), input_pos)`` — a
        numerically-identical decomposition of the original text-only forward.
        Retained as the token-input entry for the ``--logits`` NLL export path.

        Args:
            tokens: (B, T) token IDs.
            input_pos: (T,) absolute positions for RoPE / KV cache.

        Returns:
            (B, T, V) float32 soft-capped logits. Sampling is performed by the
            caller (host) or a backend source transform, not in the model.
        """
        return self.prefill_from_embeds(self.embed_text(tokens), input_pos)

    # Checkpoint loading

    @staticmethod
    def from_checkpoint(
        checkpoint_dir: str,
        max_seq_len: int = 16384,
        fuse_qkv: Optional[bool] = None,
    ) -> tuple["MuseGlimmerModel", MuseGlimmerConfig]:
        """Load a consolidated Muse Glimmer checkpoint (mp_1/ format).

        Expects ``checkpoint_dir/params.json`` and ``checkpoint_dir/mp_1/model_0.pt``.

        ``fuse_qkv`` overrides the value from ``params.json`` (default fused). When
        false, the checkpoint's fused ``qkv_proj.weight`` is split by rows into
        separate q/k/v/og projections so they can be quantized independently.
        When ``config.fuse_qko`` is set, it is instead split into ``qko_proj``
        ([Q|K|OG]) + ``v_proj`` ([V]) for the CUDA per-layer-bit-width layout.
        """
        config = MuseGlimmerConfig.from_json(
            os.path.join(checkpoint_dir, "params.json")
        )
        config.max_seq_len = max_seq_len
        if fuse_qkv is not None:
            config.fuse_qkv = fuse_qkv

        with torch.device("meta"):
            model = MuseGlimmerModel(config)

        state_dict = _load_and_remap_checkpoint(checkpoint_dir)
        if config.fuse_qko:
            state_dict = _split_fused_qkv_to_qko(state_dict, config)
        elif not config.fuse_qkv:
            state_dict = _split_fused_qkv(state_dict, config)

        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        del state_dict

        runtime_prefixes = (
            ".kv_cache.",
            ".inv_freq",
            "output_multiplier",
            "logit_softcap",
            "cache_positions",
        )
        actual_missing = set(missing)
        expected = {k for k in actual_missing if any(p in k for p in runtime_prefixes)}
        extra = actual_missing - expected
        if extra:
            raise RuntimeError(f"Missing weight keys: {sorted(extra)[:10]}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {sorted(unexpected)[:10]}")

        return model, config


# Weight loading utilities


_IGNORED_PREFIXES = (
    "vision_projection.",
    "perception_emb_norm.",
)

# Checkpoint key → model key. ``{}`` = layer index placeholder.
_CKPT_KEY_MAP = {
    # Embeddings
    "tok_embeddings.weight": "embed_tokens.weight",
    # Output
    "output_norm.weight": "output_norm.weight",
    "output.weight": "lm_head.weight",
    # Per-layer attention. The consolidated checkpoint fuses Q/K/V/OG into one
    # ``qkv_proj`` weight. It is mapped directly here; for the split (fuse_qkv=
    # False) or CUDA qko (fuse_qko=True) layouts it is post-split in
    # ``from_checkpoint`` via ``_split_fused_qkv`` / ``_split_fused_qkv_to_qko``.
    "layers.{}.attention.qkv_proj_norm.weight": "layers.{}.self_attn.qkv_proj_norm.weight",
    "layers.{}.attention.qkv_proj.weight": "layers.{}.self_attn.qkv_proj.weight",
    "layers.{}.attention.o_proj.weight": "layers.{}.self_attn.o_proj.weight",
    # Per-layer MLP
    "layers.{}.feed_forward.norm.weight": "layers.{}.mlp.norm.weight",
    "layers.{}.feed_forward.fc1_weight": "layers.{}.mlp.gate_up_proj.weight",
    "layers.{}.feed_forward.fc2_weight": "layers.{}.mlp.down_proj.weight",
    # Per-layer post-norms (same name in both)
    "layers.{}.post_attn_norm.weight": "layers.{}.post_attn_norm.weight",
    "layers.{}.post_ffn_norm.weight": "layers.{}.post_ffn_norm.weight",
}

# Pre-remapping: raw checkpoint keys have variant naming for some norms.
_RAW_KEY_FIXES = [
    (".attention.qkv_proj.norm_weight", ".attention.qkv_proj_norm.weight"),
    (".attention.qkv_proj.norm.weight", ".attention.qkv_proj_norm.weight"),
    (".feed_forward.norm_weight", ".feed_forward.norm.weight"),
    ("output.norm_weight", "output_norm.weight"),
    ("output.norm.weight", "output_norm.weight"),
]


def _fix_raw_key(key: str) -> str:
    for old, new in _RAW_KEY_FIXES:
        key = key.replace(old, new)
    return key


def _ckpt_to_model_key(key: str) -> Optional[str]:
    if key.startswith(_IGNORED_PREFIXES):
        return None

    for ckpt_pat, model_pat in _CKPT_KEY_MAP.items():
        if "{}" not in ckpt_pat:
            if key == ckpt_pat:
                return model_pat
            continue
        regex = re.escape(ckpt_pat).replace(r"\{\}", r"(\d+)")
        m = re.fullmatch(regex, key)
        if m:
            return model_pat.replace("{}", m.group(1), 1)
    return None


def _load_and_remap_checkpoint(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Load mp_1/model_0.pt and remap keys to model state_dict keys."""
    ckpt_path = os.path.join(checkpoint_dir, "mp_1", "model_0.pt")
    if not os.path.exists(ckpt_path):
        available = os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir) else []
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Available: {available}. "
            f"Re-run consolidation with --produce_mp_sizes 1."
        )

    # mmap=True keeps the (~56 GB bf16) weights file-backed instead of loading
    # them all into anonymous RAM. quantize_model then pages in one tensor at a
    # time and the OS can evict the clean pages, so peak memory is the quantized
    # output (~15-20 GB) plus a small working set rather than the whole model +
    # output (~75 GB, which OOMs a 64 GB machine). The checkpoint is already
    # bf16, so .to(bf16) below is a no-op and preserves the mmap views.
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)

    # Gemma-family checkpoints store norm weights as deltas from 1.0.
    # Fold +1.0 so the model can use standard RMSNorm (weight used directly).
    _MUSE_GLIMMER_NORM_SUFFIXES = (
        ".qkv_proj_norm.weight",
        ".mlp.norm.weight",
        ".post_attn_norm.weight",
        ".post_ffn_norm.weight",
    )

    state_dict: dict[str, torch.Tensor] = {}
    skipped = 0
    for raw_key, value in raw.items():
        fixed_key = _fix_raw_key(raw_key)
        model_key = _ckpt_to_model_key(fixed_key)
        if model_key is None:
            skipped += 1
            continue
        value = value.to(dtype=torch.bfloat16)
        if any(model_key.endswith(s) for s in _MUSE_GLIMMER_NORM_SUFFIXES):
            value = value + 1.0
        state_dict[model_key] = value
    del raw

    return state_dict


def _split_fused_qkv(
    state_dict: dict[str, torch.Tensor], config: MuseGlimmerConfig
) -> dict[str, torch.Tensor]:
    """Split each fused ``self_attn.qkv_proj.weight`` into q/k/v/og projections.

    The fused weight has output rows laid out as [Q | K | V | OG] with sizes
    ``q_dim``, ``kv_dim``, ``kv_dim``, ``og_dim``. Used when the model is built
    with ``fuse_qkv=False`` so q/k/v/og can be quantized independently.
    """
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    use_o_gate = config.use_attn_o_gate

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.endswith(".self_attn.qkv_proj.weight"):
            out[key] = value
            continue
        prefix = key[: -len("qkv_proj.weight")]
        q_end = q_dim
        k_end = q_end + kv_dim
        v_end = k_end + kv_dim
        out[f"{prefix}q_proj.weight"] = value[:q_end]
        out[f"{prefix}k_proj.weight"] = value[q_end:k_end]
        out[f"{prefix}v_proj.weight"] = value[k_end:v_end]
        if use_o_gate:
            out[f"{prefix}og_proj.weight"] = value[v_end:]
    return out


def _split_fused_qkv_to_qko(
    state_dict: dict[str, torch.Tensor], config: MuseGlimmerConfig
) -> dict[str, torch.Tensor]:
    """Split each fused ``self_attn.qkv_proj.weight`` into ``qko_proj`` + ``v_proj``.

    The fused weight rows are laid out ``[Q | K | V | OG]``. This produces
    ``qko_proj`` = ``[Q | K | OG]`` (V removed from the middle) plus a standalone
    ``v_proj`` = ``[V]``, matching the CUDA ``fuse_qko`` layout so V can carry its
    own per-layer quant bit-width.
    """
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    use_o_gate = config.use_attn_o_gate

    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.endswith(".self_attn.qkv_proj.weight"):
            out[key] = value
            continue
        prefix = key[: -len("qkv_proj.weight")]
        q_end = q_dim
        k_end = q_end + kv_dim
        v_end = k_end + kv_dim
        q = value[:q_end]
        k = value[q_end:k_end]
        v = value[k_end:v_end]
        parts = [q, k]
        if use_o_gate:
            parts.append(value[v_end:])
        out[f"{prefix}qko_proj.weight"] = torch.cat(parts, dim=0)
        out[f"{prefix}v_proj.weight"] = v
    return out


def _split_gate_up(
    state_dict: dict[str, torch.Tensor], config: MuseGlimmerConfig
) -> dict[str, torch.Tensor]:
    """Split each fused ``mlp.gate_up_proj.weight`` into gate/up projections.

    The fused weight has output rows laid out as [gate | up], each of size
    ``ffn_dim``. Splitting yields the atomic (unfused) layout so gate and up are
    stored as independent weights; fusion back into ``gate_up_proj`` is then a
    load-time concern (see ``checkpoint_loader``).
    """
    ffn_dim = config.ffn_dim
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.endswith(".mlp.gate_up_proj.weight"):
            out[key] = value
            continue
        prefix = key[: -len("gate_up_proj.weight")]
        out[f"{prefix}gate_proj.weight"] = value[:ffn_dim]
        out[f"{prefix}up_proj.weight"] = value[ffn_dim:]
    return out


def load_unfused_bf16_state_dict(
    checkpoint_dir: str, max_seq_len: int = 16384
) -> tuple[dict[str, torch.Tensor], MuseGlimmerConfig]:
    """Load a consolidated Muse Glimmer checkpoint as a fully-unfused bf16 state dict.

    Returns ``(state_dict, config)`` with the consolidated checkpoint's fused
    ``qkv_proj`` split into q/k/v/og and fused ``gate_up_proj`` split into
    gate/up -- the atomic, backend-agnostic layout. Quantization and per-backend
    fusion both operate on this canonical form downstream (``quantize_and_save``
    and ``checkpoint_loader``), so no fusion layout is baked into the checkpoint.
    """
    config = MuseGlimmerConfig.from_json(os.path.join(checkpoint_dir, "params.json"))
    config.max_seq_len = max_seq_len
    state_dict = _load_and_remap_checkpoint(checkpoint_dir)
    state_dict = _split_fused_qkv(state_dict, config)
    state_dict = _split_gate_up(state_dict, config)
    return state_dict, config


# Runtime buffer materialization


def materialize_runtime_buffers(
    model: MuseGlimmerModel,
    dtype: torch.dtype,
    device: str = "cpu",
) -> None:
    """Replace meta-device buffers with real tensors.

    Called after weight loading to fill in KV caches (zeros), RoPE tables,
    and scalar constants. Only touches buffers still on the meta device.
    """
    config = model.config

    for fqn, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        parts = fqn.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        is_kv = ".kv_cache." in fqn
        target_dtype = dtype if is_kv else torch.float32
        parent.register_buffer(
            parts[-1],
            torch.zeros(buf.shape, dtype=target_dtype, device=device),
            persistent=False,
        )

    # Recompute inv_freq (zero-fill above is wrong for RoPE frequencies)
    for layer in model.layers:
        attn = layer.self_attn
        if attn.use_rope and hasattr(attn, "_compute_inv_freq"):
            attn.register_buffer(
                "inv_freq", attn._compute_inv_freq(device=device), persistent=False
            )

    model.register_buffer(
        "output_multiplier",
        torch.tensor(config.output_multiplier, device=device),
        persistent=False,
    )
    model.register_buffer(
        "logit_softcap",
        torch.tensor(config.output_soft_cap_temp, device=device),
        persistent=False,
    )
    model.register_buffer(
        "cache_positions",
        torch.arange(config.max_seq_len, dtype=torch.long, device=device),
        persistent=False,
    )
