# This code is modified from transfomrs/models/llama/modeling_llama.py
# ----------------------------------------------------------------------
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
from executorch.examples.samsung.scripts.llm.util.masking_utils import (
    create_causal_attn_mask,
)

from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.utils.deprecation import deprecate_kwarg


class LlamaAttention_ENN(LlamaAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        out_k_cache = key_states
        out_v_cache = value_states
        key_states = torch.cat([k_cache, key_states], dim=-2)
        value_states = torch.cat([v_cache, value_states], dim=-2)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weight = query_states @ key_states.transpose(-2, -1) * self.scaling
        attn_weight += attention_mask  # attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.0, train=True)
        attn_output = attn_weight @ value_states

        attn_output = attn_output.transpose(1, 2).contiguous()

        # attn_output is (batch, token, head, dim)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, out_k_cache, out_v_cache


class LlamaDecoderLayer_ENN(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention_ENN(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, out_k_cache, out_v_cache = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, out_k_cache, out_v_cache


class LlamaPreTrainedModel_ENN(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer_ENN"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer_ENN,
        "attentions": LlamaAttention_ENN,
    }


class LlamaModel_ENN(LlamaPreTrainedModel_ENN):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer_ENN(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        k_cache: tuple[torch.Tensor, ...],
        v_cache: tuple[torch.Tensor, ...],
    ) -> BaseModelOutputWithPast:
        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        causal_mask = attention_mask
        hidden_states = inputs_embeds

        output_k_cache = []
        output_v_cache = []
        for idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            hidden_states, out_k_cache, out_v_cache = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                k_cache=k_cache[
                    idx
                ],  # (max_batch_size, n_heads, max_context_length, head_dim)
                v_cache=v_cache[
                    idx
                ],  # (max_batch_size, n_heads, max_context_length, head_dim)
            )
            output_k_cache.append(out_k_cache)
            output_v_cache.append(out_v_cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, output_k_cache, output_v_cache


class LlamaForCausalLM_ENN(LlamaPreTrainedModel_ENN, GenerationMixin):
    _tied_weights_keys = {"model.lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config,
        max_context_len: int = 1024,
        max_seq_len: int = 1024,
        ar_len: int = 1,
    ):
        super().__init__(config)
        self.model = LlamaModel_ENN(config)
        self.vocab_size = config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

        self.hidden_size = config.hidden_size
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.batch_size = 1  # batch_size
        self.num_key_value_heads = config.num_key_value_heads
        # self.max_token = 256 # max_token
        self.dtype_setting = torch.float32
        self.head_dim = config.head_dim

        self.max_context_len = max_context_len
        self.max_seq_len = max_seq_len
        self.ar_len = ar_len

        self.target_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.setup_kv_cache()

    def setup_kv_cache(self):
        cache_shape = (
            self.batch_size,
            self.num_key_value_heads,  # n_heads,
            self.max_context_len - self.ar_len,
            self.head_dim,  # head_dim
        )
        k_cache_list = []
        v_cache_list = []
        for layer in range(self.num_hidden_layers):
            k = torch.zeros(
                cache_shape, dtype=self.dtype_setting, device=self.target_device
            )
            v = torch.zeros(
                cache_shape, dtype=self.dtype_setting, device=self.target_device
            )
            k_name = f"k_cache_{layer}"
            v_name = f"v_cache_{layer}"
            self.register_buffer(k_name, k)
            self.register_buffer(v_name, v)

            k_cache_list.append(getattr(self, k_name))
            v_cache_list.append(getattr(self, v_name))

        self.k_cache = tuple(k_cache_list)
        self.v_cache = tuple(v_cache_list)

    def get_metadata(self):
        return {
            "get_bos_id": self.config.bos_token_id,
            "get_eos_ids": self.config.eos_token_id,
            "get_max_seq_len": self.max_seq_len,
            "get_max_context_len": self.max_context_len,
            "get_n_layers": self.config.num_hidden_layers,
            "get_vocab_size": self.config.vocab_size,
            "use_kv_cache": True,
            "use_sdpa_with_kv_cache": False,
            "enable_dynamic_shape": False,
            "get_prefill_ar_len": self.ar_len,
            "get_head_dim": self.config.head_dim,
            "get_n_heads": self.config.num_attention_heads,
            "get_n_kv_heads": self.config.num_key_value_heads,
            "get_hidden_dim": self.config.hidden_size,
        }

    def get_example_inputs(
        self,
        num_token=1,
    ):
        input_ids = torch.randint(
            1, self.vocab_size, (self.batch_size, num_token), device=self.target_device
        )
        attention_mask = create_causal_attn_mask(
            self.batch_size,
            num_token,
            self.max_context_len,
        ).to(device=self.target_device, dtype=self.dtype_setting)
        position_ids = torch.arange(num_token, device=self.target_device)
        position_ids = position_ids.unsqueeze(0)
        hidden_states_dummy = torch.zeros(input_ids.shape + (self.hidden_size,)).to(
            self.target_device
        )
        position_embeddings = self.rotary_emb(hidden_states_dummy, position_ids)

        example_inputs = (
            input_ids,
            attention_mask,
            position_embeddings,
            self.k_cache,
            self.v_cache,
        )

        return example_inputs
