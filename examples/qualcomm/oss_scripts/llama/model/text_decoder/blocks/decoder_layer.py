# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.examples.models.llama.model_args import ModelArgs

from .attention import ATTENTION_REGISTRY, LlamaAttention
from .feed_forward import FEED_FORWARD_REGISTRY, FeedForward
from .norm import NORM_REGISTRY


DECODER_LAYER_REGISTRY: Dict[bool, Type[nn.Module]] = {}


def register_decoder_layer(is_kv_shared: bool):
    def decorator(cls: Type[nn.Module]):
        DECODER_LAYER_REGISTRY[is_kv_shared] = cls
        return cls

    return decorator


class LlamaDecoderLayer(nn.Module):

    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = config.dim
        self.attention = ATTENTION_REGISTRY.get(
            config.model_architecture, LlamaAttention
        )(
            layer_idx=layer_idx,
            config=config,
            output_new_cache_only=output_new_cache_only,
        )

        self.feed_forward = FEED_FORWARD_REGISTRY.get(
            config.model_architecture, FeedForward
        )(config, hidden_dim=config.get_intermediate_size(layer_idx))
        self.attention_norm = NORM_REGISTRY[config.norm_type](
            config.dim, eps=config.norm_eps
        )
        self.ffn_norm = (
            NORM_REGISTRY[config.norm_type](config.dim, eps=config.norm_eps)
            if config.use_ffn_norm
            else None
        )
        self.post_attention_norm = (
            torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
            if config.post_attention_norm
            else None
        )
        self.residual_multiplier = config.residual_multiplier
        self.post_ffn_norm = (
            torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
            if config.post_ffn_norm
            else None
        )

        # Per-Layer Embeddings (Currently only used by Gemma4).
        self.use_per_layer_embedding = (
            config.vocab_size_per_layer_input and config.hidden_size_per_layer_input
        )
        if self.use_per_layer_embedding:
            self.per_layer_input_gate = nn.Linear(
                config.dim, config.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                config.hidden_size_per_layer_input, config.dim, bias=False
            )
            self.layer_scalar = nn.Parameter(torch.ones(1))
            self.post_per_layer_input_norm = NORM_REGISTRY[config.norm_type](
                config.dim, eps=config.norm_eps
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
        per_layer_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.attention_norm(x)
        h, k_cache, v_cache = self.attention(
            hidden_states,
            freqs_cos,
            freqs_sin,
            atten_mask,
            k_caches,
            v_caches,
        )
        if self.post_attention_norm:
            h = self.post_attention_norm(h)
        h = (
            x + h * self.residual_multiplier
            if self.residual_multiplier is not None
            else x + h
        )
        hidden_states = hidden_states if self.ffn_norm is None else self.ffn_norm(h)
        out = self.feed_forward(hidden_states)
        if self.post_ffn_norm:
            out = self.post_ffn_norm(out)
        output = (
            h + out * self.residual_multiplier
            if self.residual_multiplier is not None
            else h + out
        )

        # PLE
        if self.use_per_layer_embedding:
            gated = F.gelu(self.per_layer_input_gate(output), approximate="tanh")
            projected = self.per_layer_projection(gated * per_layer_input)
            output = (
                output + self.post_per_layer_input_norm(projected)
            ) * self.layer_scalar

        return output, k_cache, v_cache


@register_decoder_layer(is_kv_shared=True)
class YOCOCrossDecoderLayer(LlamaDecoderLayer):
    """Cross-decoder layer (YOCO): attention reuses a donor layer's K/V and
    produces no new cache. Shares the self layer's submodules and residual /
    FFN / PLE flow, overriding only the attention call."""

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        donor_k: Optional[torch.Tensor],
        donor_v: Optional[torch.Tensor],
        per_layer_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None, None]:
        hidden_states = self.attention_norm(x)
        h, _, _ = self.attention(
            hidden_states,
            freqs_cos,
            freqs_sin,
            atten_mask,
            donor_k=donor_k,
            donor_v=donor_v,
        )
        if self.post_attention_norm:
            h = self.post_attention_norm(h)
        h = (
            x + h * self.residual_multiplier
            if self.residual_multiplier is not None
            else x + h
        )
        hidden_states = hidden_states if self.ffn_norm is None else self.ffn_norm(h)
        out = self.feed_forward(hidden_states)
        if self.post_ffn_norm:
            out = self.post_ffn_norm(out)
        output = (
            h + out * self.residual_multiplier
            if self.residual_multiplier is not None
            else h + out
        )

        # PLE
        if self.use_per_layer_embedding:
            gated = F.gelu(self.per_layer_input_gate(output), approximate="tanh")
            projected = self.per_layer_projection(gated * per_layer_input)
            output = (
                output + self.post_per_layer_input_norm(projected)
            ) * self.layer_scalar

        return output, None, None
