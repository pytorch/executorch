# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 Text Transformer.

Complete text decoder combining:
- Self-decoder (first num_hidden_layers - num_kv_shared_layers layers)
- Cross-decoder (remaining num_kv_shared_layers layers, YOCO KV sharing)
- Final normalization
- Output projection (lm_head, tied with embed_tokens)
"""

from typing import Optional, Tuple

import torch
from torch import nn

from .gemma4_config import Gemma4Config
from .gemma4_cross_decoder import Gemma4CrossDecoder
from .gemma4_self_decoder import Gemma4SelfDecoder


def scaled_tanh(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply scaled tanh for logit softcapping."""
    return scale * torch.tanh(x / scale)


class Gemma4TextModel(nn.Module):
    """Complete Gemma 4 text decoder model.

    Combines self-decoder and cross-decoder with final output projection.
    Uses tied embeddings (lm_head shares weight with embed_tokens).

    Args:
        config: Gemma4Config with model parameters
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()

        self.config = config

        # Self-decoder (layers 0 to num_self_decoder_layers-1)
        self.self_decoder = Gemma4SelfDecoder(config)

        # Cross-decoder (layers num_self_decoder_layers to num_hidden_layers-1)
        self.cross_decoder = Gemma4CrossDecoder(config)

        # Final normalization
        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Output projection (language model head)
        # Separate Linear for quantization compatibility; weights tied at load time
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Softcapping for final logits
        self.final_logit_softcapping = config.final_logit_softcapping

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
        last_valid_token_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Forward pass through the complete text model.

        Args:
            input_ids: Token IDs of shape [batch, seq_len]; multimodal positions
                are already replaced with pad_token_id upstream.
            attention_mask: (full_attention_mask, sliding_attention_mask) tuple
            inputs_embeds: Optional pre-computed embeddings for audio injection
            mm_mask: Optional precomputed multimodal float mask
                [batch, seq_len] with values:
                - 1.0: position is audio/image token, use inputs_embeds
                - 0.0: position is regular text token, keep base_embeds
            position_embeddings_global: (cos, sin) for full attention layers
            position_embeddings_local: (cos, sin) for sliding attention layers
            k_cache: Tuple of per-layer K cache tensors from previous step.
            v_cache: Tuple of per-layer V cache tensors from previous step.
            last_valid_token_pos: Position of the token to compute lm_head for,
                as a [1] long tensor (e.g. torch.tensor([pos])). None computes
                lm_head for all seq positions.

        Returns:
            Tuple of:
            - logits: Output logits of shape [batch, seq_len, vocab_size]
              ([batch, 1, vocab_size] when last_valid_token_pos is given)
            - out_k_cache: Updated per-layer K cache tensors (self-decoder layers)
            - out_v_cache: Updated per-layer V cache tensors (self-decoder layers)
        """
        # Self-decoder
        hidden_states, per_layer_inputs, shared_kv, out_k_cache, out_v_cache = (
            self.self_decoder(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                mm_mask=mm_mask,
                attention_mask=attention_mask,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                k_cache=k_cache,
                v_cache=v_cache,
            )
        )

        # Cross-decoder (with shared K/V from self-decoder for YOCO)
        hidden_states = self.cross_decoder(
            hidden_states=hidden_states,
            per_layer_inputs=per_layer_inputs,
            shared_kv=shared_kv,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=attention_mask,
        )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Only compute lm_head for the output token at last_valid_token_pos
        if last_valid_token_pos is not None:
            hidden_states = hidden_states[:, last_valid_token_pos, :]

        # Output projection
        logits = self.lm_head(hidden_states)

        # Apply softcapping if configured
        if self.final_logit_softcapping > 0:
            logits = scaled_tanh(logits, self.final_logit_softcapping)

        return logits, out_k_cache, out_v_cache


class Gemma4ForCausalLM(nn.Module):
    """Wrapper for causal language modeling.

    Args:
        config: Gemma4Config with model parameters
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4TextModel(config)

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
        last_valid_token_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Forward pass for causal LM.

        Returns:
            Tuple of:
            - logits: Output logits
            - output_k_cache: Updated K cache tuple (one tensor per self-decoder layer)
            - output_v_cache: Updated V cache tuple (one tensor per self-decoder layer)
        """
        logits, output_k_cache, output_v_cache = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            mm_mask=mm_mask,
            attention_mask=attention_mask,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            k_cache=k_cache,
            v_cache=v_cache,
            last_valid_token_pos=last_valid_token_pos,
        )

        return logits, output_k_cache, output_v_cache
