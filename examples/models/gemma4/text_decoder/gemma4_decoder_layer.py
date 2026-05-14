# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 Decoder Layer.

Simplified from Gemma 3N: no AltUp, no LAUREL.
Adds layer_scalar for residual scaling.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .gemma4_attention import Gemma4Attention
from .gemma4_config import Gemma4Config
from .gemma4_norm import RMSNorm


class Gemma4MLP(nn.Module):
    """Gemma 4 MLP with GELU-tanh activation.

    Uses gate-up pattern: down(gelu(gate(x)) * up(x))

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Intermediate dimension
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class Gemma4DecoderLayer(nn.Module):
    """Gemma 4 Decoder Layer.

    Components:
    - Attention with QKV norms and partial RoPE
    - MLP with GELU-tanh
    - Per-layer input processing
    - Layer scalar for residual scaling

    Args:
        config: Gemma4Config with model parameters
        layer_idx: Index of this layer
    """

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = Gemma4Attention(
            config=config,
            layer_idx=layer_idx,
        )

        # MLP (cross-decoder layers use 2x intermediate_size when use_double_wide_mlp)
        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.get_intermediate_size(layer_idx),
        )

        # Layer scalar (learnable, loaded from checkpoint)
        self.layer_scalar = nn.Parameter(torch.ones(1))

        # Per-layer input processing
        self.per_layer_input_gate = nn.Linear(
            config.hidden_size,
            config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input,
            config.hidden_size,
            bias=False,
        )

        # LayerNorms
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_per_layer_input_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Residual clamping for fp16 stability
        self.use_res_clamp = config.use_res_clamp

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for decoder layer.

        Args:
            hidden_states: Input of shape [batch, seq_len, hidden_size]
            per_layer_input: Per-layer input of shape [batch, seq_len, hidden_size_per_layer_input]
            input_pos: Current position(s) for KV cache
            shared_kv: Optional tuple of (k, v) from donor layer for YOCO

        Returns:
            Tuple of (hidden_states, per_layer_input, kv_to_share)
        """
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_to_share = self.self_attn(
            hidden_states=hidden_states,
            input_pos=input_pos,
            shared_kv=shared_kv,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.use_res_clamp:
            hidden_states = torch.clamp(
                hidden_states,
                torch.finfo(torch.float16).min,
                torch.finfo(torch.float16).max,
            )

        # MLP
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.use_res_clamp:
            hidden_states = torch.clamp(
                hidden_states,
                torch.finfo(torch.float16).min,
                torch.finfo(torch.float16).max,
            )

        # Per-layer input processing
        residual = hidden_states
        gated = self.per_layer_input_gate(hidden_states)
        gated = F.gelu(gated, approximate="tanh")
        gated = gated * per_layer_input
        projected = self.per_layer_projection(gated)
        projected = self.post_per_layer_input_norm(projected)
        hidden_states = residual + projected

        # Layer scalar applied to entire output (matches HF convention)
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states, per_layer_input, kv_to_share
