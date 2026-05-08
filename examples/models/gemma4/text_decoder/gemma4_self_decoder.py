# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 Self-Decoder.

The self-decoder includes:
- Token embeddings
- Per-layer embeddings (PLE)
- First num_hidden_layers - num_kv_shared_layers decoder layers
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .gemma4_config import Gemma4Config
from .gemma4_decoder_layer import Gemma4DecoderLayer
from .gemma4_norm import RMSNorm


class Gemma4SelfDecoder(nn.Module):
    """Self-decoder component of Gemma 4.

    Contains the first (num_hidden_layers - num_kv_shared_layers) layers,
    which operate with full KV computation (no sharing).

    Args:
        config: Gemma4Config with model parameters
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()

        self.config = config
        self.num_layers = config.num_self_decoder_layers

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.embed_scale = math.sqrt(config.hidden_size)

        # Per-layer embeddings (PLE)
        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
        )
        self.embed_scale_per_layer = math.sqrt(config.hidden_size_per_layer_input)

        # Projection from model hidden to per-layer space
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        # Per-layer projection normalization
        self.per_layer_projection_norm = RMSNorm(
            config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
        )

        # Scaling factors
        self.per_layer_input_scale = 1.0 / math.sqrt(2.0)
        self.per_layer_projection_scale = 1.0 / math.sqrt(config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, layer_idx=i) for i in range(self.num_layers)]
        )

    def _compute_per_layer_inputs(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-layer input embeddings.

        Combines:
        1. Per-layer token embeddings (scaled)
        2. Projection from model hidden states (scaled and normalized)

        Args:
            input_ids: Token IDs of shape [batch, seq_len] or [seq_len]
            hidden_states: Model hidden states [batch, seq_len, hidden_size]

        Returns:
            Per-layer inputs of shape [num_layers, batch, seq_len, hidden_size_per_layer_input]
        """
        # Project from model hidden and scale
        per_layer_proj = self.per_layer_model_projection(hidden_states)
        per_layer_proj = per_layer_proj * self.per_layer_projection_scale

        # Reshape projection
        if hidden_states.dim() == 3:
            batch_size, seq_len, _ = hidden_states.shape
            per_layer_proj = per_layer_proj.view(
                batch_size,
                seq_len,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            )
        else:
            seq_len = hidden_states.shape[0]
            per_layer_proj = per_layer_proj.view(
                seq_len,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            )

        # Normalize projection
        per_layer_proj = self.per_layer_projection_norm(per_layer_proj)

        # Get per-layer token embeddings
        per_layer_mask = torch.logical_and(
            input_ids >= 0, input_ids < self.config.vocab_size_per_layer_input
        )
        per_layer_tokens = torch.where(
            per_layer_mask, input_ids, torch.zeros_like(input_ids)
        )
        per_layer_embed = (
            self.embed_tokens_per_layer(per_layer_tokens) * self.embed_scale_per_layer
        )

        if hidden_states.dim() == 3:
            per_layer_embed = per_layer_embed.view(
                batch_size,
                seq_len,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            )
        else:
            per_layer_embed = per_layer_embed.view(
                seq_len,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input,
            )

        # Combine and scale
        per_layer_input = (
            per_layer_proj + per_layer_embed
        ) * self.per_layer_input_scale

        # Permute to [num_layers, batch, seq_len, hidden_size_per_layer_input]
        if hidden_states.dim() == 3:
            per_layer_input = per_layer_input.permute(2, 0, 1, 3)
        else:
            per_layer_input = per_layer_input.permute(1, 0, 2)

        return per_layer_input

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Forward pass through self-decoder.

        Args:
            input_ids: Token IDs of shape [batch, seq_len] or [seq_len]
            input_pos: Current position(s) for KV cache
            inputs_embeds: Optional pre-computed embeddings of shape
                [batch, seq_len, hidden_size] for audio injection

        Returns:
            Tuple of:
            - hidden_states: [batch, seq_len, hidden_size]
            - remaining_per_layer_inputs for cross-decoder
            - shared_kv: Dict mapping donor layer indices to (k, v) tuples
        """
        # Clamp multimodal token IDs to 0 before embedding lookup
        if inputs_embeds is not None:
            is_multimodal = (input_ids == self.config.audio_token_id) | (
                input_ids == self.config.image_token_id
            )
            safe_ids = torch.where(
                is_multimodal,
                torch.tensor(0, dtype=input_ids.dtype, device=input_ids.device),
                input_ids,
            )
        else:
            safe_ids = input_ids

        # Compute text embeddings
        base_embeds = self.embed_tokens(safe_ids)
        base_embeds = base_embeds * self.embed_scale

        # Merge multimodal embeddings at placeholder positions
        if inputs_embeds is not None:
            mm_mask = is_multimodal.unsqueeze(-1)
            base_embeds = torch.where(mm_mask, inputs_embeds, base_embeds)

        # Compute per-layer inputs (use safe_ids so multimodal positions get
        # pad token PLE, matching HF's behavior)
        per_layer_inputs = self._compute_per_layer_inputs(safe_ids, base_embeds)

        # No altup -- use base_embeds directly
        hidden_states = base_embeds

        # Collect shared K/V from donor layers for YOCO
        shared_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[i]

            hidden_states, _, kv_to_share = layer(
                hidden_states=hidden_states,
                per_layer_input=per_layer_input,
                input_pos=input_pos,
            )

            if kv_to_share is not None:
                shared_kv[i] = kv_to_share

        # Return remaining per-layer inputs for cross-decoder
        remaining_per_layer_inputs = per_layer_inputs[self.num_layers :]

        return hidden_states, remaining_per_layer_inputs, shared_kv
