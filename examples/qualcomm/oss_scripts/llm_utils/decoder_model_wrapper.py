# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Optional

import scipy
import torch
import transformers
from transformers import GenerationConfig, PretrainedConfig
from transformers.cache_utils import Cache, StaticLayer

TRANSFORMERS_VERSION = "4.53.1"


class QnnCustomStaticLayer(StaticLayer):
    """
    Using Custom Cache so model output kv cache like static_llama.
    """

    def __init__(self, max_cache_len):
        super().__init__(max_cache_len=max_cache_len)
        self.new_keys = None
        self.new_values = None

    def update(self, key_states, value_states, cache_kwargs=None):
        self.new_keys = key_states
        self.new_values = value_states
        keys = torch.cat([self.keys, key_states], dim=-2)
        values = torch.cat([self.values, value_states], dim=-2)
        return keys, values

    def get_mask_sizes(self, cache_position):
        return self.max_cache_len, 0


class QnnCustomStaticCache(Cache):
    def __init__(self, past_k_list, past_v_list, max_cache_len):
        layers = []
        for pk, pv in zip(past_k_list, past_v_list):
            layer = QnnCustomStaticLayer(max_cache_len=max_cache_len)
            layer.max_batch_size, layer.num_heads, _, layer.head_dim = pk.shape
            layer.dtype = pk.dtype
            layer.device = pk.device
            layer.keys = pk
            layer.values = pv
            layer.is_initialized = True
            layers.append(layer)
        super().__init__(layers=layers)


def save_config_to_constant_methods(
    config: PretrainedConfig,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs,
):
    # Initialize metadata with values from model config
    metadata = {
        "get_bos_id": getattr(config, "bos_token_id", None),
        "get_eos_ids": getattr(config, "eos_token_id", None),
        "get_vocab_size": getattr(config, "vocab_size", None),
        "get_max_seq_len": getattr(config, "max_position_embeddings", None),
        "get_n_layers": getattr(config, "num_hidden_layers", None),
        "use_kv_cache": getattr(generation_config, "use_cache", None),
        "use_sdpa_with_kv_cache": False,
    }

    # Safely access fields from generation_config if it exists
    if generation_config is not None:
        # Check for cache_config and its attributes
        cache_config = getattr(generation_config, "cache_config", None)
        if cache_config is not None:
            if isinstance(cache_config, dict):
                max_seq_len = cache_config.get("max_cache_len", None)
            else:
                max_seq_len = getattr(cache_config, "max_cache_len", None)
            if max_seq_len is not None:
                metadata["get_max_seq_len"] = max_seq_len

    # Combine with any additional kwargs and filter out None values
    return {k: v for k, v in {**metadata, **kwargs}.items() if v is not None}


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch._dynamo.assume_constant_result
def get_transposed_hadamard_matrix(head_dim):
    r3_weight = torch.tensor(
        scipy.linalg.hadamard(head_dim, dtype=float) / math.sqrt(head_dim),
        dtype=torch.float32,
    )
    return r3_weight.transpose(0, 1)


def _qnn_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    if getattr(module.config, "enable_spinquant_r3", False):
        r3_weight = get_transposed_hadamard_matrix(module.head_dim)
        query = torch.matmul(query, r3_weight)
        key = torch.matmul(key, r3_weight)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _qnn_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    **kwargs,
):
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    reshaped_cache_position = cache_position.view(-1, 1)
    causal_mask = kv_arange <= reshaped_cache_position
    atten_mask = torch.full((causal_mask.shape[0], kv_length), -255.0)
    atten_mask = atten_mask.masked_fill(causal_mask, 0)
    atten_mask = atten_mask[None, None, :, :].expand(batch_size, -1, -1, -1)

    return atten_mask


class QnnPrecomputedRotaryEmbedding(torch.nn.Module):
    def __init__(self, rotary_emb: torch.nn.Module, max_seq_len: int, dtype):
        super().__init__()
        positions = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            cos, sin = rotary_emb(
                torch.zeros(1, max_seq_len, 1, dtype=dtype), positions
            )
        self.register_buffer("cos_table", cos[0].to(dtype), persistent=False)
        self.register_buffer("sin_table", sin[0].to(dtype), persistent=False)

    def forward(self, x, position_ids):
        # position_ids: [batch, seq] -> gather [batch, seq, head_dim]
        flat = position_ids.reshape(-1)
        cos = self.cos_table.index_select(0, flat).view(
            *position_ids.shape, self.cos_table.shape[-1]
        )
        sin = self.sin_table.index_select(0, flat).view(
            *position_ids.shape, self.sin_table.shape[-1]
        )
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class QnnCausalLMExportableModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self._metadata = save_config_to_constant_methods(
            model.config, model.generation_config
        )
        logging.info(f"Metadata to be recorded in PTE: {self._metadata}")

        self.num_layers = self.config.num_hidden_layers
        self.num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        self.head_dim = self.config.head_dim
        self.max_seq_len = self.config.max_seq_len
        self.ar_len = self.config.ar_len
        self.past_len = self.max_seq_len - self.ar_len

        self._register_attention_mask_for_4_53()
        self._use_precomputed_rope()

    def _use_precomputed_rope(self):
        """Swap HF's in-graph RoPE for precomputed cos/sin tables (see
        QnnPrecomputedRotaryEmbedding). Keeps HF's plumbing intact -- the decoder
        still calls ``self.rotary_emb(hidden_states, position_ids)`` -- but the
        wide-dynamic-range inv_freq math now happens on the host in fp32."""
        decoder = self.model.model
        rotary_emb = getattr(decoder, "rotary_emb", None)
        if rotary_emb is None:
            logging.warning("No rotary_emb found; skipping RoPE precompute.")
            return
        # HF API does not have a way to replace, so manually replace it.
        decoder.rotary_emb = QnnPrecomputedRotaryEmbedding(
            rotary_emb, self.max_seq_len, self.model.dtype
        )
        logging.info(
            f"Replaced in-graph RoPE with precomputed tables (max_seq_len={self.max_seq_len})."
        )

    def _register_attention_mask_for_4_53(self):
        if transformers.__version__ >= TRANSFORMERS_VERSION:
            from transformers.masking_utils import AttentionMaskInterface
            from transformers.modeling_utils import AttentionInterface

            AttentionInterface.register("qnn_attention", _qnn_attention)
            AttentionMaskInterface.register("qnn_attention", _qnn_attention_mask)
            self.model.config._attn_implementation = "qnn_attention"
            self._metadata.update({"use_sdpa_with_kv_cache": False})

    def get_example_inputs(self):
        input_ids = torch.tensor([[1]] * self.ar_len, dtype=torch.int32).view(1, -1)
        # Explicit additive causal mask, matching static_llama / KVManager:
        # 0.0 == attend, large-negative == masked. Shape [B, 1, ar_len, context_len].
        atten_mask = torch.zeros(1, 1, self.ar_len, self.max_seq_len)
        input_pos = torch.zeros(1, self.ar_len, dtype=torch.int32)
        # K cache is transposed (seq last) to match static_llama:
        #   K: [B, H, head_dim, past_len]   V: [B, H, past_len, head_dim]
        past_k = [
            torch.zeros(1, self.num_kv_heads, self.head_dim, self.past_len)
            for _ in range(self.num_layers)
        ]
        past_v = [
            torch.zeros(1, self.num_kv_heads, self.past_len, self.head_dim)
            for _ in range(self.num_layers)
        ]
        return (input_ids, atten_mask, input_pos, past_k, past_v)

    def forward(self, input_ids, atten_mask, input_pos, past_k, past_v):
        # Undo the static_llama K transpose so HF sees [B, H, past_len, head_dim].
        past_k_hf = [k.transpose(-1, -2) for k in past_k]
        cache = QnnCustomStaticCache(past_k_hf, past_v, max_cache_len=self.max_seq_len)
        cache_position = input_pos.view(-1).to(torch.long)
        outs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=atten_mask,
            past_key_values=cache,
            use_cache=True,
        )
        # Return only the new slice, transposing K back to static_llama layout.
        new_k = [layer.new_keys.transpose(-1, -2) for layer in cache.layers]
        new_v = [layer.new_values for layer in cache.layers]
        return outs.logits, new_k, new_v

    def get_metadata(self):
        return self._metadata
