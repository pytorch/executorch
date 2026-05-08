# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# pyre-unsafe
# LICENSE file in the root directory of this source tree.

"""
Gemma 4 Cross-Decoder.

The cross-decoder uses KV sharing (YOCO - You Only Cache Once) where
the last num_kv_shared_layers share KV cache with earlier layers.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .gemma4_config import Gemma4Config
from .gemma4_decoder_layer import Gemma4DecoderLayer


class Gemma4CrossDecoder(nn.Module):
    """Cross-decoder component of Gemma 4.

    Contains the last num_kv_shared_layers layers, which use KV sharing
    (YOCO optimization) where they share KV cache with specific earlier layers.

    Args:
        config: Gemma4Config with model parameters
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()

        self.config = config
        self.num_layers = config.num_cross_decoder_layers
        self.start_layer_idx = config.num_self_decoder_layers

        # Decoder layers (with layer indices continuing from self-decoder)
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(config, layer_idx=self.start_layer_idx + i)
                for i in range(self.num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        shared_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through cross-decoder.

        Args:
            hidden_states: Input from self-decoder [batch, seq_len, hidden_size]
            per_layer_inputs: Remaining per-layer inputs from self-decoder
            shared_kv: Dict mapping donor layer indices to (k, v) tuples
            input_pos: Current position(s) for KV cache

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        for i, layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[i]

            layer_idx = self.start_layer_idx + i
            kv_shared_layer_index = self.config.get_kv_shared_layer_index(layer_idx)
            layer_shared_kv = None
            if kv_shared_layer_index is not None and kv_shared_layer_index in shared_kv:
                layer_shared_kv = shared_kv[kv_shared_layer_index]

            hidden_states, _, _ = layer(
                hidden_states=hidden_states,
                per_layer_input=per_layer_input,
                input_pos=input_pos,
                shared_kv=layer_shared_kv,
            )

        return hidden_states
