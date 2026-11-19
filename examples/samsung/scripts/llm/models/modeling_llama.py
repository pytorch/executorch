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
from typing import Optional, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # transformers dispatches the weights per `device_map` and never forwards
        # it to __init__, so move the KV cache to a requested torch.device here.
        device_map = kwargs.get("device_map")
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        if isinstance(device_map, torch.device):
            model.target_device = device_map
            model.setup_kv_cache()
        return model

    def __init__(
        self,
        config,
        max_context_len: int = 1024,
        max_seq_len: int = 1024,
        ar_len: int = 1,
        device: Optional[torch.device] = None,
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

        self.target_device = device if device is not None else torch.device("cpu")
        self.setup_kv_cache()

    def setup_kv_cache(self):
        cache_shape = (
            self.batch_size,
            self.num_key_value_heads,  # n_heads,
            self.max_context_len,
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

    def _build_attention_mask(
        self, num_token: int, cache_position: torch.Tensor
    ) -> torch.Tensor:
        """Build the causal attention mask for one static ENN step.

        Mirrors the reference ENN forward: the ``num_token`` current keys are
        concatenated after ``max_context_len - num_token`` cached-history columns
        to span ``max_context_len``. The mask is derived from the absolute
        ``cache_position`` of the current tokens, so a decode step attends the
        already-cached history (columns before ``cache_position[0]``) instead of
        only its own token. Attended cells are encoded as 0 and masked cells as
        -255 so they can be added to the attention logits directly.
        """
        max_token = self.max_context_len
        history_len = max_token - num_token
        device = self.target_device
        dtype = self.dtype_setting

        cache_position = cache_position.to(device)
        # Global position of each key column: the leading history columns map to
        # their own index, the trailing current columns map to cache_position.
        col_position = torch.cat(
            [torch.arange(history_len, device=device), cache_position]
        ).view(
            1, -1
        )  # [1, max_token]
        query_position = cache_position.view(-1, 1)  # [num_token, 1]
        is_history = torch.arange(max_token, device=device) < history_len
        start = cache_position[0]

        attend = (is_history & (col_position < start)) | (
            ~is_history & (col_position <= query_position)
        )
        mask = torch.where(
            attend,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), -255.0, dtype=dtype, device=device),
        )
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_token, max_token]
        return mask.expand(self.batch_size, 1, num_token, max_token)

    def get_example_inputs(
        self,
        num_token=1,
    ):
        input_ids = torch.randint(
            1, self.vocab_size, (self.batch_size, num_token), device=self.target_device
        )
        cache_position = torch.arange(num_token, device=self.target_device)
        attention_mask = self._build_attention_mask(num_token, cache_position).to(
            self.target_device
        )
        position_ids = torch.arange(num_token, device=self.target_device)
        position_ids = position_ids.unsqueeze(0)
        hidden_states_dummy = torch.zeros(input_ids.shape + (self.hidden_size,)).to(
            self.target_device
        )
        position_embeddings = self.rotary_emb(hidden_states_dummy, position_ids)
        position_embeddings = tuple(
            data.to(self.dtype_setting) for data in position_embeddings
        )

        # Match the reference: keep the full static buffer but only feed the
        # graph a prefix of length max_context_len - num_token, so the model
        # concatenates it with the num_token new keys to span max_context_len.
        k_cache = tuple(
            k[:, :, 0 : self.max_context_len - num_token, :].to(self.target_device)
            for k in self.k_cache
        )
        v_cache = tuple(
            v[:, :, 0 : self.max_context_len - num_token, :].to(self.target_device)
            for v in self.v_cache
        )

        example_inputs = (
            input_ids,
            attention_mask,
            position_embeddings,
            k_cache,
            v_cache,
        )

        token_dim = torch.export.Dim.AUTO(min=1, max=self.max_seq_len)
        cache_dim = torch.export.Dim.AUTO(min=1, max=self.max_seq_len)
        max_token_dim = torch.export.Dim.AUTO(min=1, max=self.max_seq_len)

        kv_cache_dyn_list = []
        for _ in range(self.config.num_hidden_layers):
            kv_cache_dyn_list.append({2: cache_dim})
        kv_cache_dyn = tuple(kv_cache_dyn_list)

        dynamic_shapes = {
            "input_ids": {1: token_dim},
            "attention_mask": {2: token_dim, 3: max_token_dim},
            "position_embeddings": (
                {1: token_dim},
                {1: token_dim},
            ),
            "k_cache": kv_cache_dyn,
            "v_cache": kv_cache_dyn,
        }

        return example_inputs, dynamic_shapes

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Run a single prefill/decode step of the static ENN model.

        Builds the position embeddings and attention masks for ``input_ids``
        (typically a prefill of ``ar_len`` tokens or a single decode token, but
        any length up to ``self.max_context_len`` is accepted) and runs them
        through the static KV cache in ``self.model``. The attention mask width
        is the static cache length (``max_context_len``), of which the leading
        ``max_context_len - num_token`` entries are enabled/masked, and the newly
        produced key/value states are written back
        into those static buffers at ``cache_position`` so the history accumulates
        across decode steps. ``cache_position``/``position_ids`` are tracked with
        a ``DynamicCache`` so the model also works with HF ``generate``. The ENN
        attention masks are always reconstructed here, so any HF-style
        ``attention_mask`` argument is ignored. Extra HF ``generate`` kwargs are
        also accepted and ignored.
        """
        num_token = input_ids.shape[1]

        # Manage the KV cache sequence position for HF `generate`.
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + num_token,
                device=self.target_device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create position embeddings shared across the decoder layers.
        hidden_states_dummy = torch.zeros(
            input_ids.shape + (self.hidden_size,), device=self.target_device
        )
        position_embeddings = self.rotary_emb(hidden_states_dummy, position_ids)
        position_embeddings = tuple(
            data.to(self.dtype_setting) for data in position_embeddings
        )

        # Update past_key_values with dummy states so HF `generate` tracks the
        # sequence length, matching the reference forward.
        kv_expand = (num_token,)
        for layer_idx in range(self.config.num_hidden_layers):
            key_states_dummy = torch.zeros(
                (self.batch_size, self.config.num_key_value_heads)
                + kv_expand
                + (self.config.head_dim,)
            ).to(self.target_device)
            value_states_dummy = torch.zeros_like(key_states_dummy).to(
                self.target_device
            )
            cos, sin = position_embeddings
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states_dummy, value_states_dummy = past_key_values.update(
                key_states_dummy,
                value_states_dummy,
                layer_idx,
                cache_kwargs,
            )

        # The model concatenates the static cache prefix
        # (max_context_len - num_token) with the num_token new keys, spanning
        # max_context_len, so the attention mask must span that total. The mask
        # is derived from the absolute ``cache_position`` (like the reference
        # ENN forward) so a decode step can attend the cached history rather
        # than only its own token.
        attention_mask = self._build_attention_mask(num_token, cache_position)

        k_cache = tuple(
            k[:, :, 0 : self.max_context_len - num_token, :] for k in self.k_cache
        )
        v_cache = tuple(
            v[:, :, 0 : self.max_context_len - num_token, :] for v in self.v_cache
        )

        logits, cur_key_state_list, cur_value_state_list = self.model(
            input_ids,
            attention_mask,
            position_embeddings,
            k_cache,
            v_cache,
        )

        # Keep the KV cache managed: write the produced states back into the
        # static buffers so history accumulates across decode steps.
        for i in range(len(cur_key_state_list)):
            torch.ops.aten.index_put_(
                self.k_cache[i], [None, None, cache_position], cur_key_state_list[i]
            )
            torch.ops.aten.index_put_(
                self.v_cache[i], [None, None, cache_position], cur_value_state_list[i]
            )

        logits = logits[:, slice(-logits_to_keep, None), :]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )
