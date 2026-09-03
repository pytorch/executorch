# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DFlash draft model for Muse Glimmer speculative decoding.

The draft uses target hidden states as injected attention context and shares
the target token embeddings and output head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.examples.models.muse_glimmer.model.model import (
    FlatKVCache,
    RingKVCache,
    RMSNorm,
    ROPE_SAFE_DTYPES,
)


def rotate_half(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    allow_low_precision: bool = False,
) -> torch.Tensor:
    """Apply half-split rotary embeddings to ``[B, H, T, D]`` input.

    Draft GGUF tensors use the NEOX layout while target tensors remain
    interleaved, so the two models require different rotations.
    """
    if allow_low_precision and x.dtype in ROPE_SAFE_DTYPES:
        compute_dtype = x.dtype
    else:
        compute_dtype = torch.float32
    xc = x.to(compute_dtype)
    cos = cos.to(compute_dtype)
    sin = sin.to(compute_dtype)
    half = xc.shape[-1] // 2
    x1 = xc[..., :half]
    x2 = xc[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


@dataclass
class DFlashConfig:
    dim: int = 6656
    n_layers: int = 5
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    ffn_dim: int = 19968
    vocab_size: int = 202048
    block_size: int = 16
    target_layers: list[int] = field(default_factory=lambda: [2, 14, 26, 38, 50])
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    mask_token_id: int = 201818
    max_seq_len: int = 131072
    sliding_window: int | None = None
    sliding_window_pattern: list[bool] | None = None

    @property
    def n_target_layers(self) -> int:
        return len(self.target_layers)

    @staticmethod
    def from_gguf_metadata(metadata: dict) -> DFlashConfig:
        return DFlashConfig(
            dim=metadata.get("dflash.embedding_length", 6656),
            n_layers=metadata.get("dflash.block_count", 5),
            n_heads=metadata.get("dflash.attention.head_count", 32),
            n_kv_heads=metadata.get("dflash.attention.head_count_kv", 8),
            head_dim=metadata.get("dflash.attention.key_length", 128),
            ffn_dim=metadata.get("dflash.feed_forward_length", 19968),
            block_size=metadata.get("dflash.block_size", 16),
            target_layers=metadata.get("dflash.target_layers", [2, 14, 26, 38, 50]),
            rope_theta=metadata.get("dflash.rope.freq_base", 500000.0),
            norm_eps=metadata.get("dflash.attention.layer_norm_rms_epsilon", 1e-5),
            mask_token_id=metadata.get("tokenizer.ggml.mask_token_id", 201818),
            max_seq_len=metadata.get("dflash.context_length", 131072),
            sliding_window=metadata.get("dflash.attention.sliding_window", None),
            sliding_window_pattern=metadata.get(
                "dflash.attention.sliding_window_pattern", None
            ),
        )


def build_dflash_swa_mask(
    total_ctx: int,
    block_len: int,
    buf_size: int,
    window_size: int,
) -> torch.Tensor:
    """Build boolean validity mask for ring-buffer context + block K/V.

    The SWA ring buffer evicts entries beyond ``window_size``. This mask
    marks which ring slots are filled — uniform for all Q positions (the
    draft model's cross-attention is non-causal).  Block entries are always
    valid.

    Returns [1, 1, 1, buf_size + block_len] boolean mask (True = attend).
    """
    slots = torch.arange(buf_size, dtype=torch.long)
    ring_pos = slots + ((total_ctx - 1 - slots) // buf_size) * buf_size
    valid = (ring_pos >= 0) & (ring_pos >= total_ctx - window_size)
    blk_valid = torch.ones(block_len, dtype=torch.bool)
    return torch.cat([valid, blk_valid]).unsqueeze(0).unsqueeze(0).unsqueeze(0)


class DFlashAttention(nn.Module):
    """Non-causal cross-attention via KV injection with context KV cache.

    Q comes from the block tokens only. K/V are the concatenation of
    cached context (from projected target hidden states) and block tokens —
    the same k_proj/v_proj weights are used for both.

    Context K/V are cached: only new target hidden states are projected each
    call. The cache is populated incrementally via ``input_pos[0]`` which
    gives the write offset (ctx_start_pos).
    """

    def __init__(
        self,
        config: DFlashConfig,
        max_context_length: int = 4096,
        is_sliding: bool = False,
        window_size: int | None = None,
    ):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.dim, q_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
        self.rope_theta = config.rope_theta
        self.is_sliding = is_sliding
        self.window_size = window_size

        if is_sliding:
            self.kv_cache = RingKVCache(
                max_batch_size=1,
                window_size=window_size,
                num_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            self.kv_cache = FlatKVCache(
                max_batch_size=1,
                max_seq_len=max_context_length,
                num_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
            )

    def forward(
        self,
        x: torch.Tensor,
        target_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx_start: int,
        total_ctx: int,
    ) -> torch.Tensor:
        """
        Args:
            x: block token hidden states [B, T, D]
            target_hidden: NEW projected target hidden [B, n_new, D]
            cos, sin: RoPE frequencies [1, 1, total_ctx+T, head_dim//2]
            ctx_start: cache write offset (number of already-cached positions)
            total_ctx: total context length after adding new (ctx_start + n_new)
        """
        B, T, _ = x.shape
        n_new = target_hidden.shape[1]

        # Project new context K/V, apply k_norm, RoPE, then cache
        k_new = self.k_proj(target_hidden).view(
            B, n_new, self.n_kv_heads, self.head_dim
        )
        v_new = self.v_proj(target_hidden).view(
            B, n_new, self.n_kv_heads, self.head_dim
        )
        k_new = self.k_norm(k_new)
        k_new = k_new.transpose(1, 2)
        v_new = v_new.transpose(1, 2)
        k_new = rotate_half(
            k_new,
            cos[:, :, ctx_start:total_ctx, :],
            sin[:, :, ctx_start:total_ctx, :],
        )
        positions = torch.arange(ctx_start, total_ctx, device=k_new.device)
        k_cached, v_cached = self.kv_cache.update(positions, k_new, v_new)

        if self.is_sliding:
            attn_mask = build_dflash_swa_mask(
                total_ctx,
                T,
                self.kv_cache.buf_size,
                self.window_size,
            )
            k_ctx = k_cached
            v_ctx = v_cached
        else:
            k_ctx = k_cached[:, :, :total_ctx, :]
            v_ctx = v_cached[:, :, :total_ctx, :]
            attn_mask = None

        # Block K/V — computed fresh each call
        k_blk = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v_blk = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        k_blk = self.k_norm(k_blk)
        k_blk = k_blk.transpose(1, 2)
        v_blk = v_blk.transpose(1, 2)
        k_blk = rotate_half(
            k_blk,
            cos[:, :, total_ctx : total_ctx + T, :],
            sin[:, :, total_ctx : total_ctx + T, :],
        )

        # Q from block tokens
        xq = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        xq = self.q_norm(xq)
        xq = xq.transpose(1, 2)
        xq = rotate_half(
            xq,
            cos[:, :, total_ctx : total_ctx + T, :],
            sin[:, :, total_ctx : total_ctx + T, :],
        )

        # SDPA over cached context + block
        k = torch.cat([k_ctx, k_blk], dim=2)
        v = torch.cat([v_ctx, v_blk], dim=2)

        y = F.scaled_dot_product_attention(
            xq,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=True,
            scale=self.attn_scale,
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(y)


class DFlashMLP(nn.Module):
    """SwiGLU MLP with separate gate/up projections."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    """DFlash decoder layer: pre-norm attention + pre-norm MLP."""

    def __init__(
        self,
        config: DFlashConfig,
        max_context_length: int = 4096,
        is_sliding: bool = False,
        window_size: int | None = None,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.self_attn = DFlashAttention(
            config, max_context_length, is_sliding, window_size
        )
        self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.mlp = DFlashMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        target_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx_start: int,
        total_ctx: int,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x), target_hidden, cos, sin, ctx_start, total_ctx
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DFlashDraftModel(nn.Module):
    """DFlash draft model for block-parallel speculative decoding.

    Generates block_size tokens in a single forward pass. Does NOT own
    embed_tokens or lm_head — those are shared from the target model.

    Each layer has a per-layer KV cache for context (projected target hidden
    states). Only new target hidden states are projected each call.

    Forward:
        noise_embedding: [B, T, D] — embedded [anchor, MASK, ..., MASK]
        target_hidden: [B, n_new, N*D] — NEW target hidden states only
        input_pos: [1] — ctx_start_pos (cache write offset)
    Returns:
        hidden: [B, T, D] — pass through target's lm_head to get logits
    """

    def __init__(self, config: DFlashConfig, max_context_length: int = 4096):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(config.n_target_layers * config.dim, config.dim, bias=False)
        self.hidden_norm = RMSNorm(config.dim, config.norm_eps)

        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    config,
                    max_context_length,
                    is_sliding=(
                        config.sliding_window_pattern[i]
                        if config.sliding_window_pattern is not None
                        else False
                    ),
                    window_size=config.sliding_window,
                )
                for i in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(config.dim, config.norm_eps)

        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.hidden_norm(self.fc(target_hidden))
        n_new = ctx.shape[1]
        T = noise_embedding.shape[1]
        ctx_start = input_pos[0].item()
        total_ctx = ctx_start + n_new

        # RoPE frequencies for total_ctx + T positions
        total_len = total_ctx + T
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    device=input_pos.device,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )
        freqs = torch.outer(
            torch.arange(total_len, device=input_pos.device, dtype=torch.float32),
            inv_freq,
        )
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

        x = noise_embedding
        for layer in self.layers:
            x = layer(x, ctx, cos, sin, ctx_start, total_ctx)

        return self.norm(x)


class MuseGlimmerWithDFlash(nn.Module):
    """Wrapper combining target and draft models with shared weights.

    The target's embed_tokens and lm_head are shared with the draft. The ET
    emitter deduplicates them by content hash when both methods are exported.
    """

    def __init__(
        self, target: nn.Module, draft: nn.Module, target_config, draft_config
    ):
        super().__init__()
        self.target = target
        self.draft = draft
        self.target_config = target_config
        self.draft_config = draft_config

    def embed_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed target tokens using the backend-transformed target contract."""
        return self.target.embed_text(tokens)

    def target_forward_from_embeddings(
        self, inputs_embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the hidden-tapped target from precomputed text/image embeddings."""
        return self.target.target_forward_from_embeddings(inputs_embeds, input_pos)

    def target_prefill_from_embeddings(
        self, inputs_embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run long hidden-tapped target prefill from text/image embeddings."""
        return self.target.target_prefill_from_embeddings(inputs_embeds, input_pos)

    def target_forward(
        self, tokens: torch.Tensor, input_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible token-input hidden-tapped target forward."""
        return self.target(tokens, input_pos)

    def draft_forward(
        self,
        draft_tokens: torch.Tensor,
        target_hidden: torch.Tensor,
        input_pos: torch.Tensor,
        valid_ctx_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise_embedding = self.target.embed_tokens(draft_tokens).to(
            self.target.activation_dtype
        )

        # valid_ctx_len is a CUDA-graph-only signal (fixed-shape padded context +
        # a runtime valid length); the CUDA draft forward accepts it, while the
        # eager reference and MLX forwards keep the 3-arg signature.
        if valid_ctx_len is None:
            h = self.draft(noise_embedding, target_hidden, input_pos)
        else:
            h = self.draft(noise_embedding, target_hidden, input_pos, valid_ctx_len)

        return self.target.lm_head(h).float()

    def draft_prefill(
        self,
        draft_tokens: torch.Tensor,
        target_hidden: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Populate the Draft context cache without projecting logits."""
        noise_embedding = self.target.embed_tokens(draft_tokens).to(
            self.target.activation_dtype
        )
        return self.draft(noise_embedding, target_hidden, input_pos)
