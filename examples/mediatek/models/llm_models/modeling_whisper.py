# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from models.llm_models.configuration_whisper import (
    WhisperConfig,
    WhisperDecoderConfig,
    WhisperEncoderConfig,
)

from models.llm_models.modeling_common import Attention, ModelChunk, RMSNorm, TorchGelu
from torch import nn
from torch.export import Dim

np.random.seed(42)

# flake8: noqa: C901


class WhisperMLP(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.gelu = TorchGelu()

    def forward(self, x):
        up = self.up_proj(x)
        pre_down = self.gelu(up)
        down = self.down_proj(pre_down)

        return down


class WhisperEncoderAttention(nn.Module):
    def __init__(self, config: WhisperEncoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def forward(
        self,
        hidden_states,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            (self.q_proj(hidden_states) * self.scale)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(self.num_heads, q_len, self.head_dim)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(self.num_heads, q_len, self.head_dim)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(self.num_heads, q_len, self.head_dim)
        )

        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)

        return attn_output


class WhisperDecoderAttention(Attention):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states,
        mask,
        past_key,
        past_value,
    ):
        bsz, q_len, _ = hidden_states.size()
        c_len = past_key.size()[2]

        query_states = (
            (self.q_proj(hidden_states) * self.scale)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)
        key_states_out = key_states
        value_states_out = value_states
        if self.num_key_value_groups > 1:
            key_states = self.repeat_kv(
                key_states, bsz, q_len + c_len, self.num_key_value_groups
            )
            value_states = self.repeat_kv(
                value_states, bsz, q_len + c_len, self.num_key_value_groups
            )
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        attn_weights = attn_weights + mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)

        key_states_out = key_states_out[:, :, q_len:, :]
        value_states_out = value_states_out[:, :, q_len:, :]

        return attn_output, key_states_out, value_states_out


class WhisperCrossAttention(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states, cross_key, cross_value):
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            (self.q_proj(hidden_states) * self.scale)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        key_states = cross_key
        value_states = cross_value

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output


class WhisperEncoderLayer(nn.Module):
    def __init__(
        self,
        config: WhisperEncoderConfig,
        jit_trace=False,
        attn_class=WhisperEncoderAttention,
        mlp_class=WhisperMLP,
    ):
        super().__init__()
        self.config = config
        self.jit_trace = jit_trace
        self.hidden_size = config.hidden_size
        self.self_attn = attn_class(config)
        self.mlp = mlp_class(config)

        norm_class = RMSNorm if config.norm == "RMSNorm" else nn.LayerNorm
        self.input_norm = norm_class(self.hidden_size, eps=config.norm_eps).float()
        self.post_attention_norm = norm_class(
            self.hidden_size, eps=config.norm_eps
        ).float()

    def forward(
        self,
        hidden_states,
    ):
        residual = hidden_states

        if self.jit_trace:
            hidden_states = self.input_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.input_norm(hidden_states.to(torch.float32)).to(dtype)

        layer_device = hidden_states.device

        attn_outputs = self.self_attn(hidden_states.to(layer_device))
        hidden_states = residual.to(layer_device) + attn_outputs

        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.post_attention_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_attention_norm(
                hidden_states.to(torch.float32)
            ).to(dtype)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoderKVLayer(nn.Module):
    def __init__(self, config: WhisperEncoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()

        k_out = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_out = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        return k_out, v_out


class WhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        return_attn=False,
        jit_trace=False,
        attn_class=WhisperDecoderAttention,
        mlp_class=WhisperMLP,
        cross_attn_class=WhisperCrossAttention,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.return_attn = return_attn
        self.jit_trace = jit_trace
        self.self_attn = attn_class(config)
        self.cross_attn = cross_attn_class(config)
        self.mlp = mlp_class(config)

        norm_class = RMSNorm if config.norm == "RMSNorm" else nn.LayerNorm
        self.input_norm = norm_class(self.hidden_size, eps=config.norm_eps).float()
        self.post_attention_norm = norm_class(
            self.hidden_size, eps=config.norm_eps
        ).float()
        self.post_cross_attention_norm = norm_class(
            self.hidden_size, eps=config.norm_eps
        ).float()

    def forward(
        self,
        hidden_states,
        mask,
        past_key,
        past_value,
        cross_key,
        cross_value,
    ):
        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.input_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.input_norm(hidden_states.to(torch.float32)).to(dtype)

        layer_device = hidden_states.device

        # Self Attention
        attn_output, present_key, present_value = self.self_attn(
            hidden_states=hidden_states.to(layer_device),
            mask=mask.to(layer_device),
            past_key=past_key.to(layer_device),
            past_value=past_value.to(layer_device),
        )
        hidden_states = residual.to(layer_device) + attn_output
        residual = hidden_states

        if self.jit_trace:
            hidden_states = self.post_attention_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_attention_norm(
                hidden_states.to(torch.float32)
            ).to(dtype)

        # Cross Attention
        hidden_states = self.cross_attn(
            hidden_states=hidden_states.to(layer_device),
            cross_key=cross_key.to(layer_device),
            cross_value=cross_value.to(layer_device),
        )
        hidden_states = residual.to(layer_device) + hidden_states
        residual = hidden_states
        if self.jit_trace:
            hidden_states = self.post_cross_attention_norm(hidden_states)
        else:
            dtype = hidden_states.dtype
            hidden_states = self.post_cross_attention_norm(
                hidden_states.to(torch.float32)
            ).to(dtype)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.return_attn:
            return hidden_states, present_key, present_value, attn_output
        return hidden_states, present_key, present_value


class WhisperEncoderModel(nn.Module):
    def __init__(
        self,
        config: WhisperEncoderConfig,
        dtype=torch.float32,
        jit_trace=False,
        encoder_class=WhisperEncoderLayer,
    ):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins, config.hidden_size, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1
        )
        self.gelu1 = TorchGelu()
        self.gelu2 = TorchGelu()
        self.embed_positions = nn.Embedding(
            config.max_source_positions, config.hidden_size
        )
        self.num_layers = config.num_hidden_layers
        self.dtype = dtype
        self.jit_trace = jit_trace
        self.head_dim = config.head_dim

        self.layers = nn.ModuleList(
            [encoder_class(config, self.jit_trace) for _ in range(self.num_layers)]
        )

        norm_class = RMSNorm if config.norm == "RMSNorm" else nn.LayerNorm
        self.norm = norm_class(config.hidden_size, eps=config.norm_eps).float()
        self.encoder_tails_kv = nn.ModuleList(
            [WhisperEncoderKVLayer(config) for _ in range(config.decoder_num_layers)]
        )

    def forward(self, input_embeds):
        input_embeds = self.gelu1(self.conv1(input_embeds))
        input_embeds = self.gelu2(self.conv2(input_embeds)).permute(0, 2, 1)
        embed_pos = self.embed_positions.weight
        input_embeds = input_embeds + embed_pos.reshape(
            (1, embed_pos.shape[0], embed_pos.shape[1])
        )

        hidden_states = input_embeds
        for idx, encoder_layer in enumerate(self.layers):
            encoder_outputs = encoder_layer(hidden_states.to(self.device_list[idx]))
            hidden_states = encoder_outputs

        cross_key_cache = []
        cross_value_cache = []
        if self.jit_trace:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states.to(torch.float32)).to(self.dtype)

        for idx, encoder_tails_kv_layer in enumerate(self.encoder_tails_kv):
            k_out, v_out = encoder_tails_kv_layer(
                hidden_states.to(self.device_list[idx])
            )
            cross_key_cache.append(k_out.to(input_embeds.device))
            cross_value_cache.append(v_out.to(input_embeds.device))

        k_out = torch.cat([k for k in cross_key_cache], dim=0)
        v_out = torch.cat([v for v in cross_value_cache], dim=0)
        cross_cache = torch.cat([k_out, v_out], dim=0)

        return hidden_states, cross_cache

    def load_weights(self, state_dict):
        state_dict_start_idx = 0
        if state_dict is None:
            fake_weights = True
        else:
            input_norm_subkey = "self_attn_layer_norm"
            post_attention_norm_subkey = "final_layer_norm"
            prefix = "model.encoder."
            fake_weights = False

        outer_layer_idx = state_dict_start_idx
        self.device_list = []
        temp_state_dict = {}

        for inner_layer_idx in range(self.num_layers):
            if fake_weights:
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.q_proj.weight": torch.rand(
                            self.config.head_dim * self.config.num_attention_heads,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.k_proj.weight": torch.rand(
                            self.config.num_key_value_heads * self.head_dim,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.v_proj.weight": torch.rand(
                            self.config.num_key_value_heads * self.head_dim,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.q_proj.bias": torch.zeros(
                            self.config.head_dim * self.config.num_attention_heads,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.k_proj.bias": torch.zeros(
                            self.config.num_key_value_heads * self.head_dim,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.v_proj.bias": torch.zeros(
                            self.config.num_key_value_heads * self.head_dim,
                            dtype=self.dtype,
                        ),
                    },
                }
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.o_proj.weight": torch.rand(
                            self.config.hidden_size,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": torch.rand(
                            self.config.hidden_size,
                            self.config.intermediate_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.weight": torch.rand(
                            self.config.intermediate_size,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.input_norm.weight": torch.rand(
                            self.config.hidden_size, dtype=torch.float32
                        ),
                        f"layers.{inner_layer_idx}.post_attention_norm.weight": torch.rand(
                            self.config.hidden_size, dtype=torch.float32
                        ),
                        f"layers.{inner_layer_idx}.self_attn.o_proj.bias": torch.zeros(
                            self.config.hidden_size, dtype=self.dtype
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": torch.zeros(
                            self.config.hidden_size, dtype=self.dtype
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.bias": torch.zeros(
                            self.config.intermediate_size, dtype=self.dtype
                        ),
                    },
                }

                if self.config.norm == "LayerNorm":
                    temp_state_dict = {
                        **temp_state_dict,
                        **{
                            f"layers.{inner_layer_idx}.input_norm.bias": torch.zeros(
                                self.config.hidden_size, dtype=torch.float32
                            ),
                            f"layers.{inner_layer_idx}.post_attention_norm.bias": torch.zeros(
                                self.config.hidden_size, dtype=torch.float32
                            ),
                        },
                    }

            # Not Fake Weights
            else:
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.q_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.q_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.self_attn.k_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.k_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.self_attn.v_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.v_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.self_attn.q_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.q_proj.bias",
                            torch.zeros(
                                self.config.head_dim * self.config.num_attention_heads,
                                dtype=self.dtype,
                            ),
                        ),
                        f"layers.{inner_layer_idx}.self_attn.k_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.k_proj.bias",
                            torch.zeros(
                                self.config.num_key_value_heads * self.config.head_dim,
                                dtype=self.dtype,
                            ),
                        ),
                        f"layers.{inner_layer_idx}.self_attn.v_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.v_proj.bias",
                            torch.zeros(
                                self.config.num_key_value_heads * self.config.head_dim,
                                dtype=self.dtype,
                            ),
                        ),
                    },
                }

                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.o_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.out_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.fc2.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.fc1.weight"
                        ),
                        f"layers.{inner_layer_idx}.input_norm.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.{input_norm_subkey}.weight"
                        ).to(torch.float32),
                        f"layers.{inner_layer_idx}.post_attention_norm.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.{post_attention_norm_subkey}.weight"
                        ).to(
                            torch.float32
                        ),
                        f"layers.{inner_layer_idx}.self_attn.o_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.out_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.fc2.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.fc1.bias",
                            torch.zeros(
                                self.config.intermediate_size, dtype=self.dtype
                            ),
                        ),
                    },
                }

                if self.config.norm == "LayerNorm":
                    temp_state_dict = {
                        **temp_state_dict,
                        **{
                            f"layers.{inner_layer_idx}.input_norm.bias": state_dict.pop(
                                f"{prefix}layers.{outer_layer_idx}.{input_norm_subkey}.bias",
                                torch.zeros(self.config.hidden_size, dtype=self.dtype),
                            ).to(torch.float32),
                            f"layers.{inner_layer_idx}.post_attention_norm.bias": state_dict.pop(
                                f"{prefix}layers.{outer_layer_idx}.{post_attention_norm_subkey}.bias",
                                torch.zeros(self.config.hidden_size, dtype=self.dtype),
                            ).to(
                                torch.float32
                            ),
                        },
                    }

            if torch.cuda.device_count() == 0 or self.jit_trace:
                self.device_list.append("cpu")
            else:
                device_id = outer_layer_idx // (
                    self.config.num_hidden_layers // torch.cuda.device_count()
                    + (self.config.num_hidden_layers % torch.cuda.device_count() != 0)
                )
                self.device_list.append(f"cuda:{device_id}")
            outer_layer_idx += 1

        for dec_idx in range(self.config.decoder_num_layers):
            if fake_weights:
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"encoder_tails_kv.{dec_idx}.k_proj.weight": torch.rand(
                            self.config.num_key_value_heads * self.head_dim,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"encoder_tails_kv.{dec_idx}.k_proj.bias": torch.rand(
                            self.config.hidden_size, dtype=self.dtype
                        ),
                        f"encoder_tails_kv.{dec_idx}.v_proj.weight": torch.rand(
                            self.config.num_key_value_heads * self.head_dim,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"encoder_tails_kv.{dec_idx}.v_proj.bias": torch.rand(
                            self.config.hidden_size, dtype=self.dtype
                        ),
                    },
                }

            else:
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"encoder_tails_kv.{dec_idx}.k_proj.weight": state_dict.pop(
                            f"model.decoder.layers.{dec_idx}.encoder_attn.k_proj.weight"
                        ).to(torch.float32),
                        f"encoder_tails_kv.{dec_idx}.k_proj.bias": state_dict.pop(
                            f"model.decoder.layers.{dec_idx}.encoder_attn.k_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                        f"encoder_tails_kv.{dec_idx}.v_proj.weight": state_dict.pop(
                            f"model.decoder.layers.{dec_idx}.encoder_attn.v_proj.weight"
                        ).to(torch.float32),
                        f"encoder_tails_kv.{dec_idx}.v_proj.bias": state_dict.pop(
                            f"model.decoder.layers.{dec_idx}.encoder_attn.v_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                    },
                }

        if fake_weights:
            temp_state_dict = {
                **temp_state_dict,
                **{
                    "norm.weight": torch.rand(
                        self.config.hidden_size, dtype=torch.float32
                    ),
                    "conv1.weight": torch.rand(
                        self.config.hidden_size,
                        self.config.num_mel_bins,
                        3,
                        dtype=self.dtype,
                    ),
                    "conv2.weight": torch.rand(
                        self.config.hidden_size,
                        self.config.num_mel_bins,
                        3,
                        dtype=self.dtype,
                    ),
                    "conv1.bias": torch.zeros(
                        self.config.hidden_size, dtype=self.dtype
                    ),
                    "conv2.bias": torch.zeros(
                        self.config.hidden_size, dtype=self.dtype
                    ),
                    "embed_positions.weight": torch.rand(
                        self.config.max_source_positions,
                        self.config.hidden_size,
                        dtype=self.dtype,
                    ),
                },
            }
            if self.config.norm == "LayerNorm":
                temp_state_dict["norm.bias"] = torch.zeros(
                    self.config.hidden_size, dtype=torch.float32
                )
        else:
            temp_state_dict = {
                **temp_state_dict,
                **{
                    "norm.weight": state_dict.pop(f"{prefix}layer_norm.weight").to(
                        torch.float32
                    ),
                    "conv1.weight": state_dict.pop(f"{prefix}conv1.weight").to(
                        torch.float32
                    ),
                    "conv2.weight": state_dict.pop(f"{prefix}conv2.weight").to(
                        torch.float32
                    ),
                    "conv1.bias": state_dict.pop(
                        f"{prefix}conv1.bias",
                        torch.zeros(self.config.hidden_size, dtype=self.dtype),
                    ),
                    "conv2.bias": state_dict.pop(
                        f"{prefix}conv2.bias",
                        torch.zeros(self.config.hidden_size, dtype=self.dtype),
                    ),
                    "embed_positions.weight": state_dict.pop(
                        f"{prefix}embed_positions.weight"
                    ).to(torch.float32),
                },
            }
            if self.config.norm == "LayerNorm":
                temp_state_dict["norm.bias"] = state_dict.pop(
                    f"{prefix}layer_norm.bias",
                    torch.zeros(self.config.hidden_size, dtype=self.dtype),
                ).to(torch.float32)

        print(f"Loading weights for encoder")
        if temp_state_dict.keys() != self.state_dict().keys():
            temp_state_dict_only_keys = [
                x for x in temp_state_dict.keys() if x not in self.state_dict().keys()
            ]
            model_only_keys = [
                x for x in self.state_dict().keys() if x not in temp_state_dict.keys()
            ]
            raise RuntimeError(
                f"model state dict keys don't match with state_dict to load into model.\nModel only keys:{model_only_keys}\nstate_dict only keys:{temp_state_dict_only_keys}"
            )
        self.load_state_dict(temp_state_dict)
        for i in range(self.num_layers):
            self.layers[i].to(self.device_list[i])
        self.norm.to(self.device_list[-1])
        for i in range(self.config.decoder_num_layers):
            self.encoder_tails_kv[i].to(self.device_list[i])
        self.conv1.to(self.device_list[0])
        self.conv2.to(self.device_list[0])
        self.embed_positions.to(self.device_list[0])
        self.eval()

        return self

    def get_example_inputs(self, num_mel_bins: int = 128, cache_size: int = 512):
        example_inputs = (
            torch.randn(1, num_mel_bins, 3000, device="cpu", dtype=torch.float32),
        )

        return example_inputs


class WhisperDecoderModelChunk(ModelChunk):
    def __init__(
        self,
        config: WhisperDecoderConfig,
        num_blocks,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        return_attn=False,
        jit_trace=False,
    ):
        super().__init__(
            config,
            num_blocks,
            chunk_idx,
            dtype,
            include_tail,
            return_attn,
            jit_trace,
            decoder_class=WhisperDecoderLayer,
        )

    def forward(self, inputs_embeds, mask, pos_emb, cross_cache, *cache):
        if not self.jit_trace:
            assert (
                len(cache) == 2 * self.num_blocks
            ), f"split cache wrong number of input caches: {len(cache)} != 2*{self.num_blocks}"
            assert (
                cache[0].shape[0] == inputs_embeds.size()[0]
            ), f"split cache batch size mismatch: {cache[0].shape[0]} != {inputs_embeds.size()[0]}"

        inputs_embeds = inputs_embeds.to(self.device_list[0])

        if self.config.use_stable_embedding and self.chunk_idx == 0:
            if self.jit_trace:
                inputs_embeds = self.embed_layer_norm(inputs_embeds)
            else:
                inputs_embeds = self.embed_layer_norm(
                    inputs_embeds.to(torch.float32)
                ).to(self.dtype)

        if self.chunk_idx == 0:
            inputs_embeds = inputs_embeds + pos_emb.to(self.device_list[0])
        hidden_states = inputs_embeds

        cross_cache = [*torch.split(cross_cache, 1, dim=0)]

        next_key_cache = []
        next_value_cache = []
        if self.return_attn:
            attn_outputs = []

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            decoder_outputs = decoder_layer(
                hidden_states.to(self.device_list[idx]),
                mask=mask.to(self.device_list[idx]),
                past_key=cache[idx].to(self.device_list[idx]),
                past_value=cache[self.num_blocks + idx].to(self.device_list[idx]),
                cross_key=cross_cache[idx].to(self.device_list[idx]),
                cross_value=cross_cache[self.num_blocks + idx].to(
                    self.device_list[idx]
                ),
            )

            hidden_states = decoder_outputs[0]
            next_key_cache.append(decoder_outputs[1].to(inputs_embeds.device))
            next_value_cache.append(decoder_outputs[2].to(inputs_embeds.device))
            if self.return_attn:
                attn_outputs.append(decoder_outputs[3].to(inputs_embeds.device))

        if self.include_tail:
            if self.jit_trace:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = self.norm(hidden_states.to(torch.float32)).to(
                    self.dtype
                )
            hidden_states = self.lm_head(hidden_states)

        if self.return_attn:
            return hidden_states, *next_key_cache, *next_value_cache, *attn_outputs
        return hidden_states, *next_key_cache, *next_value_cache

    def load_weights(self, state_dict, state_dict_start_idx):
        if state_dict is None:
            fake_weights = True
        else:
            input_norm_subkey = "self_attn_layer_norm"
            post_attention_norm_subkey = "encoder_attn_layer_norm"
            prefix = "model.decoder."
            fake_weights = False

        outer_layer_idx = state_dict_start_idx
        self.device_list = []

        if self.config.use_stable_embedding and self.chunk_idx == 0:
            if fake_weights:
                temp_state_dict = {
                    "embed_layer_norm.weight": torch.rand(
                        self.config.hidden_size, dtype=torch.float32
                    ),
                    "embed_layer_norm.bias": torch.zeros(
                        self.config.hidden_size, dtype=torch.float32
                    ),
                }
            else:
                temp_state_dict = {
                    "embed_layer_norm.weight": state_dict.pop(
                        f"{prefix}embed_layer_norm.weight"
                    ).to(torch.float32),
                    "embed_layer_norm.bias": state_dict.pop(
                        f"{prefix}embed_layer_norm.bias",
                        torch.zeros(self.config.hidden_size, dtype=self.dtype),
                    ).to(torch.float32),
                }
        else:
            temp_state_dict = {}

        for inner_layer_idx in range(self.num_blocks):
            if fake_weights:
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.q_proj.weight": torch.rand(
                            self.config.head_dim * self.config.num_attention_heads,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.k_proj.weight": torch.rand(
                            self.config.num_key_value_heads * self.head_dim,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.v_proj.weight": torch.rand(
                            self.config.num_key_value_heads * self.head_dim,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.q_proj.bias": torch.zeros(
                            self.config.head_dim * self.config.num_attention_heads,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.k_proj.bias": torch.zeros(
                            self.config.num_key_value_heads * self.head_dim,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.self_attn.v_proj.bias": torch.zeros(
                            self.config.num_key_value_heads * self.head_dim,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.cross_attn.q_proj.weight": torch.rand(
                            self.config.num_attention_heads * self.head_dim,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.cross_attn.q_proj.bias": torch.zeros(
                            self.config.num_attention_heads * self.head_dim,
                            dtype=self.dtype,
                        ),
                    },
                }
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.o_proj.weight": torch.rand(
                            self.config.hidden_size,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.cross_attn.o_proj.weight": torch.rand(
                            self.config.hidden_size,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": torch.rand(
                            self.config.hidden_size,
                            self.config.intermediate_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.weight": torch.rand(
                            self.config.intermediate_size,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.input_norm.weight": torch.rand(
                            self.config.hidden_size, dtype=torch.float32
                        ),
                        f"layers.{inner_layer_idx}.post_attention_norm.weight": torch.rand(
                            self.config.hidden_size, dtype=torch.float32
                        ),
                        f"layers.{inner_layer_idx}.post_cross_attention_norm.weight": torch.rand(
                            self.config.hidden_size, dtype=torch.float32
                        ),
                        f"layers.{inner_layer_idx}.self_attn.o_proj.bias": torch.zeros(
                            self.config.hidden_size, dtype=self.dtype
                        ),
                        f"layers.{inner_layer_idx}.cross_attn.o_proj.bias": torch.zeros(
                            self.config.hidden_size, dtype=self.dtype
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": torch.zeros(
                            self.config.hidden_size, dtype=self.dtype
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.bias": torch.zeros(
                            self.config.intermediate_size, dtype=self.dtype
                        ),
                    },
                }

                if self.config.norm == "LayerNorm":
                    temp_state_dict = {
                        **temp_state_dict,
                        **{
                            f"layers.{inner_layer_idx}.input_norm.bias": torch.zeros(
                                self.config.hidden_size, dtype=torch.float32
                            ),
                            f"layers.{inner_layer_idx}.post_attention_norm.bias": torch.zeros(
                                self.config.hidden_size, dtype=torch.float32
                            ),
                            f"layers.{inner_layer_idx}.post_cross_attention_norm.bias": torch.zeros(
                                self.config.hidden_size, dtype=torch.float32
                            ),
                        },
                    }

            # Not Fake Weights
            else:
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.q_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.q_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.self_attn.k_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.k_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.self_attn.v_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.v_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.self_attn.q_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.q_proj.bias",
                            torch.zeros(
                                self.config.head_dim * self.config.num_attention_heads,
                                dtype=self.dtype,
                            ),
                        ),
                        f"layers.{inner_layer_idx}.self_attn.k_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.k_proj.bias",
                            torch.zeros(
                                self.config.num_key_value_heads * self.config.head_dim,
                                dtype=self.dtype,
                            ),
                        ),
                        f"layers.{inner_layer_idx}.self_attn.v_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.v_proj.bias",
                            torch.zeros(
                                self.config.num_key_value_heads * self.config.head_dim,
                                dtype=self.dtype,
                            ),
                        ),
                        f"layers.{inner_layer_idx}.cross_attn.q_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.encoder_attn.q_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.cross_attn.q_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.encoder_attn.q_proj.bias",
                            torch.zeros(
                                self.config.head_dim * self.config.num_attention_heads,
                                dtype=self.dtype,
                            ),
                        ),
                    },
                }

                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.o_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.out_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.cross_attn.o_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.encoder_attn.out_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.fc2.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.fc1.weight"
                        ),
                        f"layers.{inner_layer_idx}.input_norm.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.{input_norm_subkey}.weight"
                        ).to(torch.float32),
                        f"layers.{inner_layer_idx}.post_attention_norm.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.{post_attention_norm_subkey}.weight"
                        ).to(
                            torch.float32
                        ),
                        f"layers.{inner_layer_idx}.post_cross_attention_norm.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.final_layer_norm.weight"
                        ).to(
                            torch.float32
                        ),
                        f"layers.{inner_layer_idx}.self_attn.o_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.out_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                        f"layers.{inner_layer_idx}.cross_attn.o_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.encoder_attn.out_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.fc2.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.fc1.bias",
                            torch.zeros(
                                self.config.intermediate_size, dtype=self.dtype
                            ),
                        ),
                    },
                }

                if self.config.norm == "LayerNorm":
                    temp_state_dict = {
                        **temp_state_dict,
                        **{
                            f"layers.{inner_layer_idx}.input_norm.bias": state_dict.pop(
                                f"{prefix}layers.{outer_layer_idx}.{input_norm_subkey}.bias",
                                torch.zeros(self.config.hidden_size, dtype=self.dtype),
                            ).to(torch.float32),
                            f"layers.{inner_layer_idx}.post_attention_norm.bias": state_dict.pop(
                                f"{prefix}layers.{outer_layer_idx}.{post_attention_norm_subkey}.bias",
                                torch.zeros(self.config.hidden_size, dtype=self.dtype),
                            ).to(
                                torch.float32
                            ),
                            f"layers.{inner_layer_idx}.post_cross_attention_norm.bias": state_dict.pop(
                                f"{prefix}layers.{outer_layer_idx}.final_layer_norm.bias",
                                torch.zeros(self.config.hidden_size, dtype=self.dtype),
                            ).to(
                                torch.float32
                            ),
                        },
                    }

            if torch.cuda.device_count() == 0 or self.jit_trace:
                self.device_list.append("cpu")
            else:
                device_id = outer_layer_idx // (
                    self.config.num_hidden_layers // torch.cuda.device_count()
                    + (self.config.num_hidden_layers % torch.cuda.device_count() != 0)
                )
                self.device_list.append(f"cuda:{device_id}")
            outer_layer_idx += 1

        if self.include_tail:
            if fake_weights:
                temp_state_dict = {
                    **temp_state_dict,
                    "norm.weight": torch.rand(
                        self.config.hidden_size, dtype=torch.float32
                    ),
                    "lm_head.weight": torch.rand(
                        self.config.vocab_size,
                        self.config.hidden_size,
                        dtype=self.dtype,
                    ),
                    "lm_head.bias": torch.zeros(
                        self.config.vocab_size, dtype=self.dtype
                    ),
                }
                if self.config.norm == "LayerNorm":
                    temp_state_dict["norm.bias"] = torch.zeros(
                        self.config.hidden_size, dtype=torch.float32
                    )
            else:
                if self.config.tie_word_embeddings:
                    lm_head_weight_key = f"{prefix}embed_tokens.weight"
                    lm_head_bias_key = f"{prefix}embed_tokens.bias"
                else:
                    lm_head_weight_key = "lm_head.weight"
                    lm_head_bias_key = "lm_head.bias"
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        "lm_head.weight": state_dict.pop(lm_head_weight_key),
                        "norm.weight": state_dict.pop(f"{prefix}layer_norm.weight").to(
                            torch.float32
                        ),
                        "lm_head.bias": state_dict.pop(
                            lm_head_bias_key,
                            torch.zeros(self.config.vocab_size, dtype=self.dtype),
                        ),
                    },
                }
                if self.config.norm == "LayerNorm":
                    temp_state_dict["norm.bias"] = state_dict.pop(
                        f"{prefix}layer_norm.bias",
                        torch.zeros(self.config.hidden_size, dtype=self.dtype),
                    ).to(torch.float32)

        print(f"Loading weights for chunk {self.chunk_idx}")
        if temp_state_dict.keys() != self.state_dict().keys():
            temp_state_dict_only_keys = [
                x for x in temp_state_dict.keys() if x not in self.state_dict().keys()
            ]
            model_only_keys = [
                x for x in self.state_dict().keys() if x not in temp_state_dict.keys()
            ]
            raise RuntimeError(
                f"model state dict keys don't match with state_dict to load into model.\nModel only keys:{model_only_keys}\nstate_dict only keys:{temp_state_dict_only_keys}"
            )
        self.load_state_dict(temp_state_dict)
        for i in range(self.num_blocks):
            self.layers[i].to(self.device_list[i])
        if self.config.use_stable_embedding and self.chunk_idx == 0:
            self.embed_layer_norm.to(self.device_list[0])
        if self.include_tail:
            self.norm.to(self.device_list[-1])
            self.lm_head.to(self.device_list[-1])
        self.eval()

        return self

    def get_example_inputs(
        self, num_token: int = 128, cache_size: int = 512, get_dym_shape=False
    ):
        head_dim = int(self.head_dim)
        example_inputs = (
            torch.randn(
                1, num_token, self.config.hidden_size, device="cpu", dtype=torch.float32
            ),
            torch.randn(
                1,
                1,
                num_token,
                cache_size + num_token,
                device="cpu",
                dtype=torch.float32,
            ),
            torch.randn(
                1, num_token, self.config.hidden_size, device="cpu", dtype=torch.float32
            ),
            torch.randn(
                2 * self.num_blocks,
                self.config.num_key_value_heads,
                1500,
                head_dim,
                device="cpu",
                dtype=torch.float32,
            ),
            *[
                torch.randn(
                    1,
                    self.config.num_key_value_heads,
                    cache_size,
                    head_dim,
                    device="cpu",
                    dtype=torch.float32,
                )
                for _ in range(2 * self.num_blocks)
            ],
        )
        # Specify dims that would be dynamic during calibration
        # Note: Assume cache size fixed shape as torch dynamic shape cannot handle dim 3 being
        # combination of 2 dynamic dims
        if get_dym_shape:
            nt = Dim("num_token", max=num_token)
            cache_dims = tuple(({} for _ in range(2 * self.num_blocks)))
            dynamic_shapes = (
                {0: Dim.STATIC, 1: nt, 2: Dim.STATIC},
                {0: Dim.STATIC, 1: Dim.STATIC, 2: nt, 3: nt + cache_size},
                {0: Dim.STATIC, 1: nt, 2: Dim.STATIC},
                {0: Dim.STATIC, 1: Dim.STATIC, 2: Dim.STATIC, 3: Dim.STATIC},
                cache_dims,
            )
            return example_inputs, dynamic_shapes

        return example_inputs
