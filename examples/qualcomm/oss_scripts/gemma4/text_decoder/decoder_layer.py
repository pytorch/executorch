# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from executorch.examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.attention import (
    Gemma4Attention,
)
from executorch.examples.qualcomm.oss_scripts.gemma4.text_decoder.feed_forward import (
    Gemma4MLP,
)
from executorch.examples.qualcomm.oss_scripts.llama.model import NORM_REGISTRY
from torch import nn


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
        output_new_cache_only: bool = True,
        enable_masked_softmax: bool = False,
        use_cache: bool = True,
    ):
        super().__init__()
        self.use_res_clamp = config.use_res_clamp

        self.attention = Gemma4Attention(
            config, layer_idx, output_new_cache_only, enable_masked_softmax
        )
        self.feed_forward = Gemma4MLP(
            config.hidden_size, config.get_intermediate_size(layer_idx)
        )
        self.layer_scalar = nn.Parameter(torch.ones(1))

        # Per-Layer Embedding projections
        self.per_layer_input_gate = nn.Linear(
            config.hidden_size, config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input, config.hidden_size, bias=False
        )

        RMSNorm = NORM_REGISTRY["rmsnorm"]
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_per_layer_input_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.use_cache = use_cache

    def _res_clamp(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_res_clamp:
            return x
        finfo = torch.finfo(torch.float16)
        return torch.clamp(x, finfo.min, finfo.max)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        donor_k: Optional[torch.Tensor] = None,
        donor_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (hidden_states, new_k, new_v); new_k/new_v are None for shared non-donor layers."""
        residual = hidden_states
        h, new_k, new_v = self.attention(
            self.input_layernorm(hidden_states),
            freqs_cos,
            freqs_sin,
            atten_mask,
            k_cache,
            v_cache,
            donor_k,
            donor_v,
        )
        hidden_states = self._res_clamp(residual + self.post_attention_layernorm(h))

        residual = hidden_states
        h = self.feed_forward(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = self._res_clamp(residual + self.post_feedforward_layernorm(h))

        # PLE block
        residual = hidden_states
        gated = F.gelu(self.per_layer_input_gate(hidden_states), approximate="tanh")
        projected = self.per_layer_projection(gated * per_layer_input)
        hidden_states = (
            residual + self.post_per_layer_input_norm(projected)
        ) * self.layer_scalar

        if self.use_cache:
            return hidden_states, new_k, new_v
        else:
            return hidden_states
