# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Samsung Electronics Co. LTD
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
        self.embed_tokens_per_layer = nn.ModuleList(
            [
                nn.Embedding(
                    config.vocab_size_per_layer_input,
                    config.hidden_size_per_layer_input,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.embed_scale_per_layer = math.sqrt(config.hidden_size_per_layer_input)

        # Projection from model hidden to per-layer space
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        # Per-layer projection normalization
        self.per_layer_projection_norm = nn.RMSNorm(
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
            input_ids: Token IDs of shape [batch, seq_len] or [seq_len]; all ids
                must be in [0, vocab_size_per_layer_input)
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

        # Get per-layer token embeddings.
        embedded_outputs = [emb(input_ids) for emb in self.embed_tokens_per_layer]
        embedded_outputs = torch.cat(embedded_outputs, -1)

        per_layer_embed = embedded_outputs * self.embed_scale_per_layer

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
        attention_mask: tuple[torch.Tensor, torch.Tensor],
        inputs_embeds: Optional[torch.Tensor] = None,
        mm_mask: Optional[torch.Tensor] = None,
        position_embeddings_global: tuple[torch.Tensor, torch.Tensor] = None,
        position_embeddings_local: tuple[torch.Tensor, torch.Tensor] = None,
        k_cache: Optional[Tuple[torch.Tensor, ...]] = None,
        v_cache: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
    ]:
        """Forward pass through self-decoder.

        Args:
            input_ids: Token IDs of shape [batch, seq_len] or [seq_len]
            attention_mask: (full_attention_mask, sliding_attention_mask) tuple
            inputs_embeds: Optional pre-computed embeddings of shape
                [batch, seq_len, hidden_size] for audio injection
            mm_mask: Optional precomputed multimodal float mask
                [batch, seq_len] with values:
                - 1.0: position is audio/image token, use inputs_embeds
                - 0.0: position is regular text token, keep base_embeds
            position_embeddings_global: (cos, sin) for full attention layers
            position_embeddings_local: (cos, sin) for sliding attention layers
            k_cache: Tuple of per-layer K cache tensors from previous step.
            v_cache: Tuple of per-layer V cache tensors from previous step.

        Returns:
            Tuple of:
            - hidden_states: [batch, seq_len, hidden_size]
            - remaining_per_layer_inputs for cross-decoder
            - shared_kv: Dict mapping donor layer indices to (k, v) tuples
            - output_k_cache: Updated K cache tuple (one tensor per self-decoder layer)
            - output_v_cache: Updated V cache tuple (one tensor per self-decoder layer)
        """
        # Multimodal token IDs are replaced with pad_token_id upstream
        # (modeling_gemma4); the backend does not support the torch.where +
        # comparison ops that HF uses for this.
        safe_ids = input_ids

        # Compute text embeddings
        base_embeds = self.embed_tokens(safe_ids)
        base_embeds = base_embeds * self.embed_scale

        # Merge multimodal embeddings at placeholder positions.
        # Equivalence: when mm_mask=1.0 → inputs_embeds; when mm_mask=0.0 → base_embeds
        if inputs_embeds is not None:
            mm_mask = mm_mask.unsqueeze(-1)
            base_embeds = base_embeds + (inputs_embeds - base_embeds) * mm_mask

        # Compute per-layer inputs (use safe_ids so multimodal positions get
        # pad token PLE, matching HF's behavior)
        per_layer_inputs = self._compute_per_layer_inputs(
            safe_ids,
            base_embeds,
        )

        # No altup -- use base_embeds directly
        hidden_states = base_embeds

        # Collect shared K/V from donor layers for YOCO
        shared_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        output_k_cache = []
        output_v_cache = []

        # Return remaining per-layer inputs for cross-decoder
        remaining_per_layer_inputs = per_layer_inputs[self.num_layers :]

        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[i]

            hidden_states, _, kv_to_share, kv_to_out = layer(
                hidden_states=hidden_states,
                per_layer_input=per_layer_input,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=attention_mask,
                k_cache=k_cache[i] if k_cache is not None else None,
                v_cache=v_cache[i] if v_cache is not None else None,
            )

            if kv_to_share is not None:
                shared_kv[i] = kv_to_share

            if kv_to_out is not None:
                out_k_cache, out_v_cache = kv_to_out
                output_k_cache.append(out_k_cache)
                output_v_cache.append(out_v_cache)

        return (
            hidden_states,
            remaining_per_layer_inputs,
            shared_kv,
            output_k_cache,
            output_v_cache,
        )
