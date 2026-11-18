# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 Attention module.

Features:
- Per-layer head_dim (global_head_dim=512 for full attention, head_dim=256 for sliding)
- Partial RoPE for full attention layers (only first 25% of dims)
- V-norm (RMSNorm without weight)
- QK-norm applied before RoPE
- MQA with num_key_value_heads=1
- KV cache for autoregressive generation
- YOCO support for KV sharing
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

from .gemma4_config import Gemma4Config


class Gemma4Attention(nn.Module):
    """Gemma 4 attention with per-layer head_dim, partial RoPE, and MQA.

    Args:
        config: Gemma4Config with model parameters
        layer_idx: Index of this layer (for attention type selection)
    """

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.get_head_dim(layer_idx)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.is_sliding = config.is_sliding_attention(layer_idx)
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.rope_theta = config.get_rope_theta(layer_idx)

        # YOCO
        self.is_kv_shared_layer = config.is_kv_shared_layer(layer_idx)
        self.kv_shared_layer_index = config.get_kv_shared_layer_index(layer_idx)
        self.is_kv_donor_layer = config.is_kv_donor_layer(layer_idx)

        # Gemma 4 uses scaling=1.0 since QK-norm handles normalization
        # Same convention as Gemma 3N (not Gemma 3 which uses query_pre_attn_scalar**-0.5)
        self.scaling = 1.0

        # QKV projections sized by per-layer head_dim
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # QKV norms sized by per-layer head_dim
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = nn.RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for MQA/GQA."""
        if self.num_key_value_groups == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1)
        return x.reshape(batch, self.num_heads, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: tuple[torch.Tensor, torch.Tensor],
        position_embeddings: torch.Tensor = None,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Forward pass for attention.

        Args:
            hidden_states: Input of shape [batch, seq_len, hidden_size]
            mask: (full_attention_mask, sliding_attention_mask) tuple; each mask is
              broadcastable to [batch, num_heads, seq_len, kv_len]
            position_embeddings: (cos, sin) tuple for this layer's attention type
            shared_kv: Optional tuple of (k, v) from donor layer for YOCO
            k_cache: K cache tensor from previous step for this layer.
            v_cache: V cache tensor from previous step for this layer.

        Returns:
            Tuple of:
            - attn_output: Output tensor of shape [batch, seq_len, hidden_size]
            - kv_to_share: Post-cat full K/V tuple for YOCO. None for non-donor
              layers, (k, v) for donor layers.
            - kv_to_out: Pre-cat (k, v) of the current step for cache update.
              None for KV-shared layers.
        """
        batch_size, seq_len, _ = hidden_states.shape

        cos, sin = position_embeddings

        # Compute Q projection
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        # For KV shared layers, use shared K/V from donor layer
        if self.is_kv_shared_layer and shared_kv is not None:
            k, v = shared_kv
            q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=1)
        else:
            # Compute K, V projections
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
                1, 2
            )

            # Apply KV norms
            k = self.k_norm(k)
            v = self.v_norm(v)

            # Apply RoPE (partial for full attention, full for sliding)
            q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=1)
            k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=1)

        # Save pre-cat k/v (matches Gemma3: return new K/V before concat)
        # The concatenated k/v is used only for attention computation internally.
        kv_to_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if not self.is_kv_shared_layer:
            kv_to_out = (k, v)

        # Cat cache (Gemma3 style) — only modifies local k/v for attention,
        # does NOT overwrite out_k_cache/out_v_cache.
        if not self.is_kv_shared_layer and k_cache is not None:
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)

        # kv_to_share: post-cat full K/V for YOCO sharing
        kv_to_share: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.is_kv_donor_layer:
            kv_to_share = (k, v)

        mask = mask[1] if self.is_sliding else mask[0]
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        attn_output = self.o_proj(attn_output)

        # === Return: kv_to_share (post-cat) for YOCO, out_k/v_cache (pre-cat) for cache update ===
        return attn_output, kv_to_share, kv_to_out
