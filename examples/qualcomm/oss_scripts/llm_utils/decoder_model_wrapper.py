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

from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

TRANSFORMERS_VERSION = "4.53.1"


def save_config_to_constant_methods(
    config: PretrainedConfig,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs,
):
    # Initialize metadata with values from model config
    metadata = {
        "get_bos_id": getattr(config, "bos_token_id", None),
        "get_eos_id": getattr(config, "eos_token_id", None),
        "get_vocab_size": getattr(config, "vocab_size", None),
        "get_max_seq_len": getattr(config, "max_position_embeddings", None),
        "use_kv_cache": getattr(generation_config, "use_cache", None),
        "use_sdpa_with_kv_cache": False,
    }

    # Safely access fields from generation_config if it exists
    if generation_config is not None:
        # Check for cache_config and its attributes
        cache_config = getattr(generation_config, "cache_config", None)
        if cache_config is not None:
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

    # Simplest and most efficient way to obtain a causal mask
    causal_mask = kv_arange <= reshaped_cache_position
    atten_mask = torch.full((causal_mask.shape[0], kv_length), torch.tensor(-65504.0))
    atten_mask = atten_mask.masked_fill(causal_mask, 0)
    atten_mask = atten_mask[None, None, :, :].expand(batch_size, -1, -1, -1)

    return atten_mask


class QnnCausalLMExportableModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self._metadata = save_config_to_constant_methods(
            model.config, model.generation_config
        )
        logging.info(f"Metadata to be recorded in PTE: {self._metadata}")
        self.exportable_module = TorchExportableModuleForDecoderOnlyLM(
            self.model,
            max_batch_size=1,
            max_cache_len=self._metadata.get("get_max_seq_len"),
        )
        self._register_attention_mask_for_4_53(self.exportable_module)

    def _register_attention_mask_for_4_53(self, exportable_module: torch.nn.Module):
        if transformers.__version__ >= TRANSFORMERS_VERSION:
            from transformers.masking_utils import AttentionMaskInterface
            from transformers.modeling_utils import AttentionInterface

            AttentionInterface.register("qnn_attention", _qnn_attention)
            AttentionMaskInterface.register("qnn_attention", _qnn_attention_mask)
            exportable_module.model.model.config._attn_implementation = "qnn_attention"
            self._metadata.update({"use_sdpa_with_kv_cache": False})

    def get_example_inputs(self):
        example_input_ids = torch.tensor([[1]], dtype=torch.long)
        example_cache_position = torch.tensor([0], dtype=torch.long)
        return (example_input_ids, example_cache_position)

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor):
        return self.exportable_module(input_ids, cache_position)

    def get_metadata(self):
        return self._metadata
