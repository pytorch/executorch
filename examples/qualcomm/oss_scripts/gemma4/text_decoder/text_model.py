# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch

from executorch.examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.cross_decoder import (
    Gemma4CrossDecoder,
)
from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.self_decoder import (
    Gemma4SelfDecoder,
)
from executorch.examples.qualcomm.oss_scripts.llama.model import NORM_REGISTRY
from torch import nn


class Gemma4TextModel(nn.Module):

    def __init__(
        self,
        config: Gemma4Config,
        output_new_cache_only: bool = True,
        enable_masked_softmax: bool = False,
    ):
        super().__init__()
        self.config = config

        self.self_decoder = Gemma4SelfDecoder(
            config, output_new_cache_only, enable_masked_softmax
        )
        self.cross_decoder = Gemma4CrossDecoder(
            config, output_new_cache_only, enable_masked_softmax
        )
        self.norm = NORM_REGISTRY["rmsnorm"](
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_softcapping = config.final_logit_softcapping

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor],
        global_freqs_cos: torch.Tensor,
        global_freqs_sin: torch.Tensor,
        local_freqs_cos: torch.Tensor,
        local_freqs_sin: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[
        torch.Tensor, List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]
    ]:
        layer_types = self.config.layer_types
        n_self = self.config.num_self_decoder_layers

        hidden_states = (
            self.self_decoder.embed_tokens(tokens) * self.self_decoder.embed_scale
        )
        per_layer_inputs = self.self_decoder.compute_per_layer_inputs(
            tokens, hidden_states
        )

        # Self-decoder layers (KV I/O)
        output_k_cache: List[Optional[torch.Tensor]] = []
        output_v_cache: List[Optional[torch.Tensor]] = []
        shared_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for i, layer in enumerate(self.self_decoder.layers):
            k_cache = k_caches[i] if k_caches else None
            v_cache = v_caches[i] if v_caches else None

            if layer_types[i] == "sliding_attention":
                fc, fs, mask = local_freqs_cos, local_freqs_sin, window_atten_mask
            else:
                fc, fs, mask = global_freqs_cos, global_freqs_sin, atten_mask

            hidden_states, new_k, new_v = layer(
                hidden_states,
                per_layer_inputs[i],
                fc,
                fs,
                mask,
                k_cache,
                v_cache,
            )
            output_k_cache.append(new_k)
            output_v_cache.append(new_v)

            # Donor layers expose their full updated K/V for YOCO sharing
            if layer.attention.is_kv_donor_layer:
                if layer.attention.output_new_cache_only and k_cache is not None:
                    full_k = torch.cat([k_cache, new_k], dim=-1)
                    full_v = torch.cat([v_cache, new_v], dim=2)
                else:
                    full_k = new_k
                    full_v = new_v
                shared_kv[i] = (full_k, full_v)

        # Cross-decoder layers (no KV I/O, consume donor K/V)
        hidden_states = self.cross_decoder(
            hidden_states,
            per_layer_inputs[n_self:],
            global_freqs_cos,
            global_freqs_sin,
            local_freqs_cos,
            local_freqs_sin,
            atten_mask,
            window_atten_mask,
            layer_types,
            shared_kv,
        )

        hidden_states = self.norm(hidden_states)
        # Compute logits only for the last token(s)
        logits = self.lm_head(hidden_states)

        if self.final_logit_softcapping > 0:
            logits = self.final_logit_softcapping * torch.tanh(
                logits / self.final_logit_softcapping
            )

        return logits, output_k_cache, output_v_cache
