# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Mapping

import torch
from torch import nn

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModelWithoutEmbedding,
    repeat_kv,
)
from executorch.examples.qualcomm.robotics.policies.smolvla.config import (
    SmolVLAConfig,
)

VLM_POLICY_PREFIX = "model.vlm_with_expert.vlm.model.text_model."
# HF SmolVLM text layer -> static_llama LlamaDecoderLayer
LAYER_WEIGHT_NAMES = {
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}


def load_layer_weights(
    layers: nn.ModuleList,
    policy_state_dict: Mapping[str, torch.Tensor],
    policy_prefix: str,
) -> None:
    """Copy HF decoder layer weights into static_llama decoder layers."""
    for layer_idx, layer in enumerate(layers):
        layer_weights = {
            target: policy_state_dict[f"{policy_prefix}layers.{layer_idx}.{source}"]
            for source, target in LAYER_WEIGHT_NAMES.items()
        }
        # Only buffers may stay unloaded; unexpected keys mean a wrong mapping.
        missing, unexpected = layer.load_state_dict(layer_weights, strict=False)
        missing = [key for key in missing if key in dict(layer.named_parameters())]
        if missing or unexpected:
            raise ValueError(
                f"{policy_prefix}layers.{layer_idx}: missing={missing}, "
                f"unexpected={unexpected}"
            )


# Masked scores are replaced by this margin below the row minimum.
MASKED_SCORE_MARGIN = -20


class PrefixAttention(LlamaAttention):
    """static_llama attention with RoPE indexed by position_ids and a float mask.

    Indexing the RoPE tables inside the graph keeps repeated positions of padding
    tokens identical to SmolVLA. The mask is 0.0 to attend and nonzero to mask,
    applied with an HTP-friendly masked softmax.
    """

    def forward_with_pos(
        self,
        hidden_states: torch.Tensor,  # (B, S, dim)
        freqs_cos_full: torch.Tensor,  # (max_context_len, head_dim // 2)
        freqs_sin_full: torch.Tensor,
        position_ids: torch.Tensor,  # (S,)
        atten_mask: torch.Tensor,  # (B, 1, S, S) float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        freqs_cos = freqs_cos_full[position_ids]
        freqs_sin = freqs_sin_full[position_ids]
        q = (
            self.wq(hidden_states)
            .view(bsz, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.wk(hidden_states)
            .view(bsz, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.wv(hidden_states)
            .view(bsz, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
        k = self.apply_rope_emb(k, freqs_cos, freqs_sin).transpose(2, 3)
        kh = repeat_kv(k, self.num_key_value_groups)
        vh = repeat_kv(v, self.num_key_value_groups)

        attn = (q @ kh) / (float(self.head_dim) ** 0.5)
        attn_min = torch.amin(attn, dim=-1, keepdim=True)
        attn = torch.where(atten_mask == 0, attn, attn_min + MASKED_SCORE_MARGIN)
        attn = self.attn_softmax(attn)
        out = (attn @ vh).transpose(1, 2).reshape(bsz, seq_len, -1)
        # Contiguous caches keep one K/V ABI between the Prefix and Action Expert PTEs.
        return self.wo(out), k.contiguous(), v.contiguous()


class PrefixDecoderLayer(LlamaDecoderLayer):
    def __init__(self, layer_idx: int, config: ModelArgs):
        super().__init__(
            layer_idx=layer_idx, config=config, output_new_cache_only=True
        )
        self.attention = PrefixAttention(
            layer_idx=layer_idx, config=config, output_new_cache_only=True
        )

    def forward_with_pos(
        self,
        x: torch.Tensor,
        freqs_cos_full: torch.Tensor,
        freqs_sin_full: torch.Tensor,
        position_ids: torch.Tensor,
        atten_mask: torch.Tensor,  # (B, S, S) float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k, v = self.attention.forward_with_pos(
            self.attention_norm(x),
            freqs_cos_full,
            freqs_sin_full,
            position_ids,
            atten_mask[:, None, :, :],
        )
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x)), k, v


class PrefixLlamaModel(LlamaModelWithoutEmbedding):
    """Self-attention decoder stack returning the compact K/V cache of every layer."""

    def __init__(self, config: ModelArgs):
        super().__init__(
            config=config,
            ar_len=1,
            output_new_cache_only=True,
            output_cache=True,
        )
        self.layers = nn.ModuleList(
            [PrefixDecoderLayer(i, config) for i in range(config.n_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, S, dim)
        atten_mask: torch.Tensor,  # (B, S, S) float
        position_ids: torch.Tensor,  # (B, S) int32
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        k_caches, v_caches = [], []
        for layer in self.layers:
            hidden_states, k, v = layer.forward_with_pos(
                hidden_states,
                self.freqs_cos,
                self.freqs_sin,
                position_ids[0],
                atten_mask,
            )
            k_caches.append(k)
            v_caches.append(v)
        return k_caches, v_caches


def make_additive_attention_mask(visibility: torch.Tensor) -> torch.Tensor:
    """0.0 where visible; the static decoder masks any nonzero entry."""
    return torch.where(visibility, 0.0, SmolVLAPrefixModel.padding_mask_value)


class SmolVLAPrefixEmbedding(nn.Module):
    """Host-side language and state embeddings appended after the camera embeddings.

    Mirrors VLAFlowMatching.embed_prefix for the non-image tokens. It runs eagerly
    on the host; the Prefix Model graph only consumes the resulting embeddings.
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        text_config = config.vlm_config.text_config
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.state_proj = nn.Linear(config.state_dim, text_config.hidden_size)

    @classmethod
    def from_hf_policy(
        cls, config: SmolVLAConfig, policy_state_dict: Mapping[str, torch.Tensor]
    ) -> "SmolVLAPrefixEmbedding":
        model = cls(config)
        model.load_state_dict(
            {
                "embed_tokens.weight": policy_state_dict[
                    f"{VLM_POLICY_PREFIX}embed_tokens.weight"
                ],
                "state_proj.weight": policy_state_dict["model.state_proj.weight"],
                "state_proj.bias": policy_state_dict["model.state_proj.bias"],
            }
        )
        return model

    def forward(
        self, language_token_ids: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        language_embeddings = self.embed_tokens(language_token_ids)
        language_embeddings = language_embeddings * math.sqrt(
            language_embeddings.shape[-1]
        )
        state_embeddings = self.state_proj(state).unsqueeze(1)
        return torch.cat((language_embeddings, state_embeddings), dim=1)


class SmolVLAPrefixModel(nn.Module):
    """VLM prefix prefill: prefix embeddings -> compact K/V cache of every layer.

    Inputs:
        prefix_embeddings  (1, P, 960) float32, camera embeddings then prefix tail
        attention_mask     (1, P, P) float32, 0.0 attends and nonzero masks
        position_ids       (1, P) int32
    Outputs:
        16 keys (1, 5, 64, P) followed by 16 values (1, 5, P, 64)
    """

    padding_mask_value = -1000.0

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        model_args = config.create_prefix_model_args()
        self.hidden_size = model_args.dim
        self.model = PrefixLlamaModel(model_args)

    @classmethod
    def from_hf_policy(
        cls, config: SmolVLAConfig, policy_state_dict: Mapping[str, torch.Tensor]
    ) -> "SmolVLAPrefixModel":
        model = cls(config)
        load_layer_weights(model.model.layers, policy_state_dict, VLM_POLICY_PREFIX)
        return model

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        prefix_length = self.config.prefix_length
        return (
            torch.zeros(1, prefix_length, self.hidden_size),
            make_additive_attention_mask(
                torch.ones(1, prefix_length, prefix_length, dtype=torch.bool)
            ),
            torch.arange(prefix_length, dtype=torch.int32).unsqueeze(0),
        )

    def forward(
        self,
        prefix_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.model(prefix_embeddings, attention_mask, position_ids)
