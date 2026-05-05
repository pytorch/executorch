# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from typing import Optional

import torch
from torch import nn

from .gemma4_config import Gemma4Config
from .gemma4_cross_decoder import Gemma4CrossDecoder
from .gemma4_norm import RMSNorm
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
        self.norm = RMSNorm(
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

        # Create shared masks (one copy each, referenced by all attention layers)
        self._share_masks(config)

    def _share_masks(self, config: Gemma4Config) -> None:
        """Create causal and sliding window masks once, share across all attention layers.

        All layers use identical masks sized [max_seq_len, max_seq_len].
        Sharing avoids duplicating these large constant tensors in the PTE.
        """
        # Causal mask (upper triangular -inf)
        causal_mask = torch.full(
            (config.max_seq_len, config.max_seq_len), float("-inf")
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Sliding window mask
        sw_mask = torch.zeros(config.max_seq_len, config.max_seq_len)
        for i in range(config.max_seq_len):
            window_start = max(0, i - config.sliding_window + 1)
            if window_start > 0:
                sw_mask[i, :window_start] = float("-inf")

        # Assign shared tensors to all attention layers
        for module in self.modules():
            if hasattr(module, "causal_mask"):
                module.causal_mask = causal_mask
                if (
                    hasattr(module, "sliding_window")
                    and module.sliding_window is not None
                ):
                    module.sliding_window_mask = sw_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the complete text model.

        Args:
            input_ids: Token IDs of shape [batch, seq_len]
            input_pos: Current position(s) for KV cache
            inputs_embeds: Optional pre-computed embeddings for audio injection

        Returns:
            logits: Output logits of shape [batch, seq_len, vocab_size]
        """
        # Self-decoder
        hidden_states, per_layer_inputs, shared_kv = self.self_decoder(
            input_ids=input_ids,
            input_pos=input_pos,
            inputs_embeds=inputs_embeds,
        )

        # Cross-decoder (with shared K/V from self-decoder for YOCO)
        hidden_states = self.cross_decoder(
            hidden_states=hidden_states,
            per_layer_inputs=per_layer_inputs,
            shared_kv=shared_kv,
            input_pos=input_pos,
        )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Only compute lm_head for the last token during prefill
        hidden_states = hidden_states[:, -1:, :]

        # Output projection
        logits = self.lm_head(hidden_states)

        # Apply softcapping if configured
        if self.final_logit_softcapping > 0:
            logits = scaled_tanh(logits, self.final_logit_softcapping)

        return logits


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
        input_pos: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for causal LM."""
        return self.model(
            input_ids=input_ids, input_pos=input_pos, inputs_embeds=inputs_embeds
        )
