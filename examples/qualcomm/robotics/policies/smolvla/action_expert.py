# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
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
from executorch.examples.qualcomm.robotics.policies.smolvla.prefix_decoder import (
    load_layer_weights,
    make_additive_attention_mask,
)

EXPERT_POLICY_PREFIX = "model.vlm_with_expert.lm_expert."
# Masked scores are replaced by this margin below the row minimum.
MASKED_SCORE_MARGIN = -20


class ExpertAttention(LlamaAttention):
    """static_llama attention over the compact (n_kv_heads) prefix K/V cache.

    Self-attention layers attend to [prefix ; own] K/V. Cross-attention layers
    project the prefix K/V with their own k/v projections and attend to it only.
    RoPE tables are indexed by position_ids and the float mask is 0.0 to attend
    and nonzero to mask, applied with an HTP-friendly masked softmax.
    """

    def forward_with_pos(
        self,
        hidden_states: torch.Tensor,  # (B, S, dim)
        freqs_cos_full: torch.Tensor,  # (max_context_len, head_dim // 2)
        freqs_sin_full: torch.Tensor,
        position_ids: torch.Tensor,  # (S,)
        atten_mask: torch.Tensor,  # (B, 1, S, KV) float
        prefix_k: torch.Tensor,  # (B, kv, head_dim, P)
        prefix_v: torch.Tensor,  # (B, kv, P, head_dim)
        cross_attn_only: bool,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        freqs_cos = freqs_cos_full[position_ids]
        freqs_sin = freqs_sin_full[position_ids]
        q = (
            self.wq(hidden_states)
            .view(bsz, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = self.apply_rope_emb(q, freqs_cos, freqs_sin)

        if cross_attn_only:
            # Head-major flattening matches the SmolVLA expert k_proj/v_proj inputs.
            _, kv_heads, head_dim, prefix_len = prefix_k.shape
            k_flat = prefix_k.permute(0, 3, 1, 2).reshape(
                bsz, prefix_len, kv_heads * head_dim
            )
            v_flat = prefix_v.permute(0, 2, 1, 3).reshape(
                bsz, prefix_len, kv_heads * head_dim
            )
            k = (
                self.wk(k_flat)
                .view(bsz, prefix_len, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
                .transpose(2, 3)
            )
            v = (
                self.wv(v_flat)
                .view(bsz, prefix_len, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
        else:
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
            # Concatenate in compact GQA space, then expand once.
            k = torch.cat(
                [prefix_k, self.apply_rope_emb(k, freqs_cos, freqs_sin).transpose(2, 3)],
                dim=-1,
            )
            v = torch.cat([prefix_v, v], dim=2)
        kh = repeat_kv(k, self.num_key_value_groups)
        vh = repeat_kv(v, self.num_key_value_groups)

        attn = (q @ kh) / (float(self.head_dim) ** 0.5)
        attn_min = torch.amin(attn, dim=-1, keepdim=True)
        attn = torch.where(atten_mask == 0, attn, attn_min + MASKED_SCORE_MARGIN)
        attn = self.attn_softmax(attn)
        out = (attn @ vh).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.wo(out)


class ExpertDecoderLayer(LlamaDecoderLayer):
    def __init__(self, layer_idx: int, config: ModelArgs):
        super().__init__(
            layer_idx=layer_idx, config=config, output_new_cache_only=True
        )
        self.attention = ExpertAttention(
            layer_idx=layer_idx, config=config, output_new_cache_only=True
        )

    def forward_with_pos(
        self,
        x: torch.Tensor,
        freqs_cos_full: torch.Tensor,
        freqs_sin_full: torch.Tensor,
        position_ids: torch.Tensor,
        atten_mask: torch.Tensor,  # (B, S, KV) float
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        cross_attn_only: bool,
    ) -> torch.Tensor:
        x = x + self.attention.forward_with_pos(
            self.attention_norm(x),
            freqs_cos_full,
            freqs_sin_full,
            position_ids,
            atten_mask[:, None, :, :],
            prefix_k,
            prefix_v,
            cross_attn_only,
        )
        return x + self.feed_forward(self.ffn_norm(x))


class ExpertLlamaModel(LlamaModelWithoutEmbedding):
    """Decoder stack alternating self-attention over [prefix ; actions] and cross-attention to the prefix."""

    def __init__(self, config: ModelArgs):
        super().__init__(
            config=config,
            ar_len=1,
            output_new_cache_only=True,
            output_cache=True,
        )
        self.layers = nn.ModuleList(
            [ExpertDecoderLayer(i, config) for i in range(config.n_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, S, expert_dim)
        atten_mask: torch.Tensor,  # (B, S, P + S) float
        position_ids: torch.Tensor,  # (B, S) int32, offset by valid prefix tokens
        cross_position_ids: torch.Tensor,  # (S,) int32, restarting from 0
        k_prefix: list[torch.Tensor],
        v_prefix: list[torch.Tensor],
        self_attn_every_n_layers: int,
    ) -> torch.Tensor:
        cross_atten_mask = atten_mask[:, :, : k_prefix[0].shape[-1]]
        for i, layer in enumerate(self.layers):
            cross = i % self_attn_every_n_layers != 0
            hidden_states = layer.forward_with_pos(
                hidden_states,
                self.freqs_cos,
                self.freqs_sin,
                cross_position_ids if cross else position_ids[0],
                cross_atten_mask if cross else atten_mask,
                k_prefix[i],
                v_prefix[i],
                cross_attn_only=cross,
            )
        return self.norm(hidden_states)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float
) -> torch.Tensor:
    """Mirrors lerobot SmolVLA create_sinusoidal_pos_embedding (float64 intermediates)."""
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    sin_input = (1.0 / period * 2 * math.pi)[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


class SmolVLAActionExpert(nn.Module):
    """One flow matching Euler step of the SmolVLA Action Expert.

    The time embedding of every Euler step of the fixed schedule is precomputed
    and selected by euler_step, so no sin/cos runs in the graph.

    Inputs:
        noisy_actions        (1, chunk, max_action_dim) float32
        euler_step           (1,) int32 in [0, num_steps)
        k_prefix[layers]     (1, 5, 64, P) Prefix Model keys
        v_prefix[layers]     (1, 5, P, 64) Prefix Model values
        attention_mask       (1, chunk, P + chunk) float32, 0.0 attends and nonzero masks
        suffix_position_ids  (1, chunk) int32, offset by the valid prefix token count
    Output:
        velocity             (1, chunk, max_action_dim) float32
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        self.expert_config = config.create_action_expert_config()
        model_args = config.create_action_expert_model_args()
        hidden_size = model_args.dim
        self.model = ExpertLlamaModel(model_args)
        # Cross-attention layers project the compact VLM K/V, not the expert hidden states.
        vlm_kv_dim = (
            self.expert_config.num_key_value_heads * self.expert_config.head_dim
        )
        for layer_idx, layer in enumerate(self.model.layers):
            if not self.expert_config.is_self_attention_layer(layer_idx):
                layer.attention.wk = nn.Linear(
                    vlm_kv_dim, layer.attention.wk.out_features, bias=False
                )
                layer.attention.wv = nn.Linear(
                    vlm_kv_dim, layer.attention.wv.out_features, bias=False
                )
        self.action_in_proj = nn.Linear(config.max_action_dim, hidden_size)
        self.action_out_proj = nn.Linear(hidden_size, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(hidden_size * 2, hidden_size)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)
        self.register_buffer(
            "cross_position_ids",
            torch.arange(config.chunk_size, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "euler_time_embeddings",
            self.make_euler_time_embeddings(),
            persistent=False,
        )

    def make_euler_time_embeddings(self) -> torch.Tensor:
        """float32 time embedding of each step, t = 1 - step / num_steps."""
        dt = -1.0 / self.config.num_steps
        euler_times = torch.tensor(
            [1.0 + step * dt for step in range(self.config.num_steps)],
            dtype=torch.float32,
        )
        return create_sinusoidal_pos_embedding(
            euler_times,
            self.action_in_proj.out_features,
            self.expert_config.min_period,
            self.expert_config.max_period,
        ).float()

    @classmethod
    def from_hf_policy(
        cls, config: SmolVLAConfig, policy_state_dict: Mapping[str, torch.Tensor]
    ) -> "SmolVLAActionExpert":
        model = cls(config)
        load_layer_weights(model.model.layers, policy_state_dict, EXPERT_POLICY_PREFIX)
        model.model.norm.load_state_dict(
            {"weight": policy_state_dict[f"{EXPERT_POLICY_PREFIX}norm.weight"]}
        )
        for name, projection in (
            ("action_in_proj", model.action_in_proj),
            ("action_out_proj", model.action_out_proj),
            ("action_time_mlp_in", model.action_time_mlp_in),
            ("action_time_mlp_out", model.action_time_mlp_out),
        ):
            projection.load_state_dict(
                {
                    "weight": policy_state_dict[f"model.{name}.weight"],
                    "bias": policy_state_dict[f"model.{name}.bias"],
                }
            )
        return model

    def get_example_inputs(self) -> tuple:
        config = self.config
        expert_config = self.expert_config
        prefix_length = config.prefix_length
        prefix_pad_mask = torch.ones(1, prefix_length, dtype=torch.bool)
        visibility = torch.cat(
            (
                prefix_pad_mask[:, None, :].expand(1, config.chunk_size, prefix_length),
                torch.tril(
                    torch.ones(config.chunk_size, config.chunk_size, dtype=torch.bool)
                ).unsqueeze(0),
            ),
            dim=2,
        )
        return (
            torch.zeros(1, config.chunk_size, config.max_action_dim),
            torch.zeros(1, dtype=torch.int32),
            [
                torch.zeros(
                    1, expert_config.num_key_value_heads, expert_config.head_dim, prefix_length
                )
                for _ in range(expert_config.num_layers)
            ],
            [
                torch.zeros(
                    1, expert_config.num_key_value_heads, prefix_length, expert_config.head_dim
                )
                for _ in range(expert_config.num_layers)
            ],
            make_additive_attention_mask(visibility),
            (
                prefix_length + torch.arange(config.chunk_size, dtype=torch.int32)
            ).unsqueeze(0),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        euler_step: torch.Tensor,
        k_prefix: list[torch.Tensor],
        v_prefix: list[torch.Tensor],
        attention_mask: torch.Tensor,
        suffix_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        action_embeddings = self.action_in_proj(noisy_actions)
        time_embeddings = torch.index_select(self.euler_time_embeddings, 0, euler_step)
        time_embeddings = time_embeddings.unsqueeze(1).expand_as(action_embeddings)
        suffix = self.action_time_mlp_out(
            F.silu(
                self.action_time_mlp_in(
                    torch.cat([action_embeddings, time_embeddings], dim=-1)
                )
            )
        )
        hidden_states = self.model(
            suffix,
            attention_mask,
            suffix_position_ids,
            self.cross_position_ids,
            k_prefix,
            v_prefix,
            self.expert_config.self_attn_every_n_layers,
        )
        return self.action_out_proj(hidden_states)
