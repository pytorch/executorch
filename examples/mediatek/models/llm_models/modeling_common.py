"""Common backbone across multiple models"""

import math

import numpy as np
import torch
from models.llm_models.configuration_base import BaseConfig
from models.llm_models.modeling_base import BaseModelChunk
from torch import nn
from torch.export import Dim

torch.manual_seed(42)
np.random.seed(42)


# flake8: noqa: C901


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class Gelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        pre_down = gate * torch.sigmoid(gate) * up
        down = self.down_proj(pre_down)

        return down


class Attention(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attn_scale = math.sqrt(self.head_dim)

        if config.combine_qkv:
            self.qkv_proj = nn.Linear(
                self.hidden_size,
                (2 * self.num_key_value_heads * self.head_dim) + self.hidden_size,
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim
            )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def apply_rotary_pos_emb_mtk(self, q, k, cos, sin):
        q1 = q[..., : q.shape[-1] // 2]
        q2 = q[..., q.shape[-1] // 2 :]
        q_rotated = torch.cat((-q2, q1), dim=-1)
        k1 = k[..., : k.shape[-1] // 2]
        k2 = k[..., k.shape[-1] // 2 :]
        k_rotated = torch.cat((-k2, k1), dim=-1)

        q_embed = q * cos + q_rotated * sin
        k_embed = k * cos + k_rotated * sin
        return q_embed, k_embed

    def repeat_kv(self, hidden_states, batch, q_len, n_rep):
        if isinstance(hidden_states, list):
            output = []
            for hs in hidden_states:
                output.append(
                    hs.repeat(1, 1, n_rep, 1).view(batch, 1, q_len, self.head_dim)
                )
            return output
        else:
            hidden_states = hidden_states.repeat(1, 1, n_rep, 1)
            return hidden_states.view(batch, self.num_heads, q_len, self.head_dim)

    def forward(
        self,
        hidden_states,  # (b, t, 4096)
        mask,  # (b, 1, t, c+t)
        pos_emb,  # (b, 2, t, head dim)
        past_key,  # (b, num kv heads, c, head dim)
        past_value,  # (b, num kv heads, c, head dim)
    ):
        bsz, q_len, _ = hidden_states.size()
        c_len = past_key.size()[2]

        if self.config.combine_qkv:
            proj = self.qkv_proj(hidden_states)
            query_states = (
                proj[:, :, : self.config.hidden_size]
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            key_states = (
                proj[
                    :,
                    :,
                    self.config.hidden_size : self.config.hidden_size
                    + self.num_key_value_heads * self.head_dim,
                ]
                .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            value_states = (
                proj[
                    :,
                    :,
                    self.config.hidden_size
                    + self.num_key_value_heads * self.head_dim :,
                ]
                .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
        else:
            query_states = (
                self.q_proj(hidden_states)
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

        if self.config.position_embedding == "rope":
            cos, sin = torch.split(pos_emb, 1, dim=1)
            query_states, key_states = self.apply_rotary_pos_emb_mtk(
                query_states, key_states, cos, sin
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
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) / self.attn_scale
        )
        attn_weights = attn_weights + mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        key_states_out = key_states_out[:, :, q_len:, :]
        value_states_out = value_states_out[:, :, q_len:, :]

        return attn_output, key_states_out, value_states_out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: BaseConfig,
        return_attn=False,
        jit_trace=False,
        attn_class=Attention,
        mlp_class=MLP,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.return_attn = return_attn
        self.jit_trace = jit_trace
        self.self_attn = attn_class(config)
        self.mlp = mlp_class(config)
        if config.norm == "RMSNorm":
            self.input_norm = RMSNorm(config.hidden_size, eps=config.norm_eps).float()
            self.post_attention_norm = RMSNorm(
                config.hidden_size, eps=config.norm_eps
            ).float()
        else:
            self.input_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps
            ).float()
            self.post_attention_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps
            ).float()

    def forward(
        self,
        hidden_states,  # (b, t, hidden_dim)
        mask,  # (b, 1, t, c+t)
        pos_emb,  # (b, 2, t, head_dim)
        past_key,  # (b, num_kv_head, c, head_dim)
        past_value,  # (b, num_kv_head, c, head_dim)
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
            pos_emb=pos_emb.to(layer_device),
            past_key=past_key.to(layer_device),
            past_value=past_value.to(layer_device),
        )
        hidden_states = residual.to(layer_device) + attn_output

        # Fully Connected
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

        if self.return_attn:
            return hidden_states, present_key, present_value, attn_output
        return hidden_states, present_key, present_value


class ModelChunk(BaseModelChunk):
    def __init__(
        self,
        config: BaseConfig,
        num_blocks,
        chunk_idx,
        dtype=torch.float32,
        include_tail=False,
        return_attn=False,
        jit_trace=False,
        decoder_class=DecoderLayer,
    ):
        super().__init__(
            config, num_blocks, chunk_idx, dtype, include_tail, return_attn, jit_trace
        )
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layers = nn.ModuleList(
            [
                decoder_class(config, return_attn=return_attn, jit_trace=jit_trace)
                for _ in range(num_blocks)
            ]
        )

        if self.config.use_stable_embedding and self.chunk_idx == 0:
            self.embed_layer_norm = nn.LayerNorm(config.hidden_size).float()

        if self.include_tail:
            if config.norm == "RMSNorm":
                self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps).float()
            else:
                self.norm = nn.LayerNorm(
                    config.hidden_size, eps=config.norm_eps
                ).float()
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, inputs_embeds, mask, pos_emb, *cache):
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

        hidden_states = inputs_embeds

        next_key_cache = []
        next_value_cache = []
        if self.return_attn:
            attn_outputs = []

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            decoder_outputs = decoder_layer(
                hidden_states.to(self.device_list[idx]),
                mask=mask.to(self.device_list[idx]),
                pos_emb=pos_emb.to(self.device_list[idx]),
                past_key=cache[idx].to(self.device_list[idx]),
                past_value=cache[self.num_blocks + idx].to(self.device_list[idx]),
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
            expected_subkey = f"layers.{state_dict_start_idx}.self_attn.o_proj.weight"
            state_dict_keys = list(state_dict.keys())
            temp_key = None
            input_norm_subkey = None
            post_attention_norm_subkey = None
            for key in state_dict_keys:
                if expected_subkey in key:
                    temp_key = key
                if (
                    f"layers.{state_dict_start_idx}" in key
                    and "norm" in key
                    and "input" in key
                ):
                    input_norm_subkey = key.split(".")[-2]
                if (
                    f"layers.{state_dict_start_idx}" in key
                    and "norm" in key
                    and "post_attention" in key
                ):
                    post_attention_norm_subkey = key.split(".")[-2]
            if temp_key is None:
                raise KeyError(
                    f"Cannot find layer {state_dict_start_idx}'s o_proj weight inside state_dict. "
                    f"Please ensure o_proj weight key contains: {expected_subkey}"
                )
            if input_norm_subkey is None:
                raise KeyError(
                    f"Cannot find layer {state_dict_start_idx}'s input norm weight inside state_dict. "
                    f"Please ensure input norm weight key contains: layers.{state_dict_start_idx}, norm, and input inside"
                    " the key string."
                )
            if post_attention_norm_subkey is None:
                raise KeyError(
                    f"Cannot find layer {state_dict_start_idx}'s post attention norm weight inside state_dict."
                    f" Please ensure post attention norm weight key contains: layers.{state_dict_start_idx}, norm, and "
                    "post_attention inside the key string."
                )
            prefix = temp_key.split(expected_subkey)[0]
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
                if self.config.combine_qkv:
                    temp_state_dict[
                        f"layers.{inner_layer_idx}.self_attn.qkv_proj.weight"
                    ] = torch.rand(
                        3 * self.config.hidden_size,
                        self.config.hidden_size,
                        dtype=self.dtype,
                    )
                    temp_state_dict[
                        f"layers.{inner_layer_idx}.self_attn.qkv_proj.bias"
                    ] = torch.zeros(
                        (2 * self.config.num_key_value_heads * self.head_dim)
                        + self.config.hidden_size,
                        dtype=self.dtype,
                    )
                else:
                    temp_state_dict = {
                        **temp_state_dict,
                        **{
                            f"layers.{inner_layer_idx}.self_attn.q_proj.weight": torch.rand(
                                self.config.hidden_size,
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
                                self.config.hidden_size, dtype=self.dtype
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
                        f"layers.{inner_layer_idx}.mlp.gate_proj.weight": torch.rand(
                            self.config.intermediate_size,
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
                        f"layers.{inner_layer_idx}.mlp.gate_proj.bias": torch.zeros(
                            self.config.intermediate_size, dtype=self.dtype
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

            else:
                if self.config.combine_qkv:
                    temp_state_dict[
                        f"layers.{inner_layer_idx}.self_attn.qkv_proj.weight"
                    ] = state_dict.pop(
                        f"{prefix}layers.{outer_layer_idx}.self_attn.qkv_proj.weight"
                    )
                    temp_state_dict[
                        f"layers.{inner_layer_idx}.self_attn.qkv_proj.bias"
                    ] = state_dict.pop(
                        f"{prefix}layers.{outer_layer_idx}.self_attn.qkv_proj.bias",
                        torch.zeros(
                            (2 * self.config.num_key_value_heads * self.head_dim)
                            + self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                    )
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
                                torch.zeros(self.config.hidden_size, dtype=self.dtype),
                            ),
                            f"layers.{inner_layer_idx}.self_attn.k_proj.bias": state_dict.pop(
                                f"{prefix}layers.{outer_layer_idx}.self_attn.k_proj.bias",
                                torch.zeros(
                                    self.config.num_key_value_heads * self.head_dim,
                                    dtype=self.dtype,
                                ),
                            ),
                            f"layers.{inner_layer_idx}.self_attn.v_proj.bias": state_dict.pop(
                                f"{prefix}layers.{outer_layer_idx}.self_attn.v_proj.bias",
                                torch.zeros(
                                    self.config.num_key_value_heads * self.head_dim,
                                    dtype=self.dtype,
                                ),
                            ),
                        },
                    }

                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        f"layers.{inner_layer_idx}.self_attn.o_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.self_attn.o_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.gate_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.gate_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.down_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.up_proj.weight"
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
                            f"{prefix}layers.{outer_layer_idx}.self_attn.o_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                        f"layers.{inner_layer_idx}.mlp.gate_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.gate_proj.bias",
                            torch.zeros(
                                self.config.intermediate_size, dtype=self.dtype
                            ),
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.down_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
                        ),
                        f"layers.{inner_layer_idx}.mlp.up_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.up_proj.bias",
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
                    lm_head_weight_key = "embed_tokens.weight"
                    lm_head_bias_key = "embed_tokens.bias"
                else:
                    lm_head_weight_key = "lm_head.weight"
                    lm_head_bias_key = "lm_head.bias"
                temp_state_dict = {
                    **temp_state_dict,
                    **{
                        "lm_head.weight": state_dict.pop(lm_head_weight_key),
                        "norm.weight": state_dict.pop(f"{prefix}norm.weight").to(
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
                        f"{prefix}norm.bias",
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
        head_dim = int(self.config.hidden_size / self.config.num_attention_heads)
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
            torch.randn(1, 2, num_token, head_dim, device="cpu", dtype=torch.float32),
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
        # Note: Assume cache size fixed shape as torch dynamic shape cannot handle dim 3 being
        # combination of 2 dynamic dims
        if get_dym_shape:
            nt = Dim("num_token", max=num_token)
            cache_dims = tuple(({} for _ in range(2 * self.num_blocks)))
            dynamic_shapes = (
                {0: None, 1: nt, 2: None},
                {0: None, 1: None, 2: nt, 3: nt + cache_size},
                {0: None, 1: None, 2: nt, 3: None},
                cache_dims,
            )
            return example_inputs, dynamic_shapes

        return example_inputs
