# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
from typing import Optional

import torch
import transformers
from executorch.examples.qualcomm.oss_scripts.llama.model.apply_rope import (
    apply_rotary_emb_single,
)
from transformers import GenerationConfig, PretrainedConfig
from transformers.cache_utils import Cache, StaticLayer

TRANSFORMERS_VERSION = "4.53.1"


class QnnCustomStaticLayer(StaticLayer):
    """StaticLayer that returns cat(past, new) to attention (so the current layer
    sees the full context, matching static_llama.py:494), and stashes the pre-cat
    new K/V slice so the wrapper can return only the new slot as graph output."""

    def __init__(self, max_cache_len):
        super().__init__(max_cache_len=max_cache_len)
        self.new_keys = None
        self.new_values = None

    def update(self, key_states, value_states, cache_kwargs=None):
        self.new_keys = key_states
        self.new_values = value_states
        keys = torch.cat([self.keys, key_states.transpose(2, 3)], dim=-1)
        values = torch.cat([self.values, value_states], dim=-2)
        return keys, values

    def get_mask_sizes(self, cache_position):
        return self.max_cache_len, 0


class QnnCustomStaticCache(Cache):
    """StaticCache-shaped cache seeded from external past K/V tensors, one pair
    per layer. `max_cache_len` is the full context length (past_len + ar_len)."""

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
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-1]]
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
    atten_mask = torch.full((causal_mask.shape[0], kv_length), -65535.0)
    atten_mask = atten_mask.masked_fill(causal_mask, 0)
    atten_mask = atten_mask[None, None, :, :].expand(batch_size, -1, -1, -1)

    return atten_mask


def _qnn_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
):
    return (
        apply_rotary_emb_single(q, cos, sin),
        apply_rotary_emb_single(k, cos, sin),
    )


class QnnPrecomputedRotaryEmbedding(torch.nn.Module):
    def __init__(self, rotary_emb: torch.nn.Module, max_seq_len: int, dtype):
        super().__init__()
        positions = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        dummy = torch.zeros(1, max_seq_len, 1, dtype=dtype)
        with torch.no_grad():
            cos, sin = rotary_emb(dummy, positions)
        half = cos.shape[-1] // 2
        self.register_buffer(
            "cos_table", cos[0][:, :half].to(dtype).contiguous(), persistent=False
        )
        self.register_buffer(
            "sin_table", sin[0][:, :half].to(dtype).contiguous(), persistent=False
        )

    def forward(self, x, position_ids):
        flat = position_ids.reshape(-1)
        cos = self.cos_table.index_select(0, flat)
        sin = self.sin_table.index_select(0, flat)
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
        decoder = self.model.model
        rotary_emb = getattr(decoder, "rotary_emb", None)
        if rotary_emb is None:
            logging.warning("No rotary_emb found; skipping RoPE precompute.")
            return
        decoder.rotary_emb = QnnPrecomputedRotaryEmbedding(
            rotary_emb, self.max_seq_len, self.model.dtype
        )

        modeling = importlib.import_module(type(decoder).__module__)
        if not hasattr(modeling, "apply_rotary_pos_emb"):
            logging.warning(
                f"{modeling.__name__} has no apply_rotary_pos_emb; "
                "keeping HF's rotation."
            )
        else:
            modeling.apply_rotary_pos_emb = _qnn_apply_rotary_pos_emb
            logging.info(
                f"Patched {modeling.__name__}.apply_rotary_pos_emb with "
                "static_llama's apply_rotary_emb_single."
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
        input_tokens = torch.ones((1, self.ar_len), dtype=torch.int32)
        # Explicit additive causal mask, matching static_llama / KVManager:
        # 0.0 == attend, large-negative == masked. Shape [B, 1, ar_len, context_len].
        atten_mask = torch.zeros(1, 1, self.ar_len, self.max_seq_len)
        pos_ids = torch.zeros((1, self.ar_len), dtype=torch.int32)
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
        return (input_tokens, atten_mask, pos_ids, past_k, past_v)

    def forward(self, input_tokens, atten_mask, pos_ids, past_k, past_v):
        cache = QnnCustomStaticCache(past_k, past_v, max_cache_len=self.max_seq_len)

        outs = self.model(
            input_ids=input_tokens,
            attention_mask=atten_mask,
            position_ids=pos_ids,
            past_key_values=cache,
            cache_position=pos_ids,
            use_cache=True,
        )
        # Return only the new slice, transposing K back to static_llama layout.
        new_k = [layer.new_keys.transpose(-1, -2) for layer in cache.layers]
        new_v = [layer.new_values for layer in cache.layers]
        return outs.logits, new_k, new_v

    def get_metadata(self):
        return self._metadata
