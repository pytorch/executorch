# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch

from executorch.examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.decoder_layer import (
    Gemma4DecoderLayer,
)
from torch import nn


class Gemma4CrossDecoder(nn.Module):
    """Cross-decoder: last 20 layers with KV-sharing"""

    def __init__(
        self,
        config: Gemma4Config,
        output_new_cache_only: bool = True,
        enable_masked_softmax: bool = False,
    ):
        super().__init__()
        self.config = config
        self.start_layer_idx = config.num_self_decoder_layers  # 35 - 20 - 15
        self.num_layers = config.num_cross_decoder_layers

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    config,
                    self.start_layer_idx + i,
                    output_new_cache_only,
                    enable_masked_softmax,
                    use_cache=False,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        global_freqs_cos: torch.Tensor,
        global_freqs_sin: torch.Tensor,
        local_freqs_cos: torch.Tensor,
        local_freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        layer_types: List[str],
        shared_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            layer_idx = self.start_layer_idx + i
            kv_src = self.config.get_kv_shared_layer_index(layer_idx)
            donor_k, donor_v = None, None
            if kv_src is not None and kv_src in shared_kv:
                donor_k, donor_v = shared_kv[kv_src]

            if layer_types[layer_idx] == "sliding_attention":
                fc, fs, mask = local_freqs_cos, local_freqs_sin, window_atten_mask
            else:
                fc, fs, mask = global_freqs_cos, global_freqs_sin, atten_mask

            hidden_states = layer(
                hidden_states,
                per_layer_inputs[i],
                fc,
                fs,
                mask,
                donor_k=donor_k,
                donor_v=donor_v,
            )

        return hidden_states
