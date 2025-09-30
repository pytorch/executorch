# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from models.llm_models.configuration_phi import PhiConfig

from models.llm_models.modeling_common import Attention, DecoderLayer, ModelChunk
from torch import nn
from torch.export import Dim

np.random.seed(42)

# flake8: noqa: C901


class PhiMLP(nn.Module):
    def __init__(self, config: PhiConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        up = self.gate_up_proj(x)
        gate, up = up.chunk(2, dim=-1)
        pre_down = gate * torch.sigmoid(gate) * up
        down = self.down_proj(pre_down)

        return down


class Phi3Attention(Attention):
    def __init__(self, config: PhiConfig, jit_trace=None):
        super().__init__(config, jit_trace)


class Phi4Attention(Attention):
    def __init__(self, config: PhiConfig, jit_trace=None):
        super().__init__(config, jit_trace)

    def apply_rotary_pos_emb_mtk(self, q, k, cos, sin):
        """Apply rotary positional embedding to query and key states.

        Phi4-mini uses partial rotary embedding, which only rotate some of the q and k.
        """

        rotary_dim = cos.shape[-1]

        # Use split
        q_rot, q_pass = torch.split(q, rotary_dim, dim=-1)
        q1, q2 = torch.split(q_rot, rotary_dim // 2, dim=-1)
        q_rotated = torch.cat((-q2, q1), dim=-1)

        k_rot, k_pass = torch.split(k, rotary_dim, dim=-1)
        k1, k2 = torch.split(k_rot, rotary_dim // 2, dim=-1)
        k_rotated = torch.cat((-k2, k1), dim=-1)

        q_embed = q_rot * cos + q_rotated * sin
        k_embed = k_rot * cos + k_rotated * sin

        q_embed = torch.cat((q_embed, q_pass), dim=-1)
        k_embed = torch.cat((k_embed, k_pass), dim=-1)

        return q_embed, k_embed


class Phi3DecoderLayer(DecoderLayer):
    def __init__(
        self,
        config: PhiConfig,
        return_attn=False,
        jit_trace=False,
    ):
        super().__init__(
            config,
            return_attn,
            jit_trace,
            attn_class=Phi3Attention,
            mlp_class=PhiMLP,
        )


class Phi4DecoderLayer(DecoderLayer):
    def __init__(
        self,
        config: PhiConfig,
        return_attn=False,
        jit_trace=False,
    ):
        super().__init__(
            config,
            return_attn,
            jit_trace,
            attn_class=Phi4Attention,
            mlp_class=PhiMLP,
        )


class Phi3ModelChunk(ModelChunk):
    def __init__(
        self,
        config: PhiConfig,
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
            decoder_class=Phi3DecoderLayer,
        )

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
                        f"layers.{inner_layer_idx}.mlp.gate_up_proj.weight": torch.rand(  # Check Again
                            2 * self.config.intermediate_size,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": torch.rand(
                            self.config.hidden_size,
                            self.config.intermediate_size,
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
                        f"layers.{inner_layer_idx}.mlp.gate_up_proj.bias": torch.zeros(
                            2 * self.config.intermediate_size, dtype=self.dtype
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": torch.zeros(
                            self.config.hidden_size, dtype=self.dtype
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
                        f"layers.{inner_layer_idx}.mlp.gate_up_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.gate_up_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.down_proj.weight"
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
                        f"layers.{inner_layer_idx}.mlp.gate_up_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.gate_up_proj.bias",
                            torch.zeros(
                                2 * self.config.intermediate_size, dtype=self.dtype
                            ),
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.down_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
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
                    lm_head_weight_key = f"{prefix}embed_tokens.weight"
                    lm_head_bias_key = f"{prefix}embed_tokens.bias"
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


class Phi4ModelChunk(ModelChunk):
    def __init__(
        self,
        config: PhiConfig,
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
            decoder_class=Phi4DecoderLayer,
        )
        self.partial_rotary_factor = config.partial_rotary_factor

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
                        f"layers.{inner_layer_idx}.mlp.gate_up_proj.weight": torch.rand(  # Check Again
                            2 * self.config.intermediate_size,
                            self.config.hidden_size,
                            dtype=self.dtype,
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": torch.rand(
                            self.config.hidden_size,
                            self.config.intermediate_size,
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
                        f"layers.{inner_layer_idx}.mlp.gate_up_proj.bias": torch.zeros(
                            2 * self.config.intermediate_size, dtype=self.dtype
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": torch.zeros(
                            self.config.hidden_size, dtype=self.dtype
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
                        f"layers.{inner_layer_idx}.mlp.gate_up_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.gate_up_proj.weight"
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.weight": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.down_proj.weight"
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
                        f"layers.{inner_layer_idx}.mlp.gate_up_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.gate_up_proj.bias",
                            torch.zeros(
                                2 * self.config.intermediate_size, dtype=self.dtype
                            ),
                        ),
                        f"layers.{inner_layer_idx}.mlp.down_proj.bias": state_dict.pop(
                            f"{prefix}layers.{outer_layer_idx}.mlp.down_proj.bias",
                            torch.zeros(self.config.hidden_size, dtype=self.dtype),
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
                    lm_head_weight_key = f"{prefix}embed_tokens.weight"
                    lm_head_bias_key = f"{prefix}embed_tokens.bias"
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
                1,
                2,
                num_token,
                int(head_dim * self.partial_rotary_factor),
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
                {0: Dim.STATIC, 1: Dim.STATIC, 2: nt, 3: Dim.STATIC},
                cache_dims,
            )
            return example_inputs, dynamic_shapes

        return example_inputs
