# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
from torch import nn
from typing import Optional, Tuple
from transformers.models.qwen2.modeling_qwen2 import repeat_kv, Qwen2Attention, Qwen2RMSNorm
from transformers.cache_utils import Cache
from transformers.models.qwen2 import modeling_qwen2
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import apply_rotary_emb_single
import scipy
import math

class R3Hadamard(nn.Linear):
    def __init__(self, head_dim: int):
        super(R3Hadamard, self).__init__(in_features=head_dim, out_features=head_dim, bias=False, dtype=torch.float)
        self.head_dim = head_dim
        self.is_initialized = False

    def initialize_r3_hadamard(self):
        r3_weight = torch.tensor(scipy.linalg.hadamard(self.head_dim) / math.sqrt(self.head_dim), dtype=torch.float)
        self.weight.data.copy_(r3_weight.T)
        self.is_initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            raise AssertionError(f"{self.__class__.__name__} has not been fully initialized. \
                                 Invoke class method `initialize_r3_hadamard()` before doing a forward pass with this object")
        return super().forward(x)
    

# Copy from transformers/models/qwen2/modeling_qwen2.py, transformers version: 4.47.1
class QCQwen2Attention(Qwen2Attention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super(QCQwen2Attention, self).__init__(config, layer_idx)

        if getattr(config, "enable_spinquant_r3", False):
            self.q_R3 = R3Hadamard(head_dim = self.head_dim)
            self.k_R3 = R3Hadamard(head_dim = self.head_dim)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)


        cos, sin = position_embeddings
        query_states = apply_rotary_emb_single(query_states, cos, sin)
        key_states = apply_rotary_emb_single(key_states, cos, sin)

        if getattr(self.config, "enable_spinquant_r3", False):
            query_states = self.q_R3(query_states)
            key_states = self.k_R3(key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)/ math.sqrt(self.head_dim)) 
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

orig_embedding_fwd = modeling_qwen2.Qwen2RotaryEmbedding.forward
def bypass_rotary_embedding(self, x, position_ids):
    if isinstance(position_ids, tuple) and len(position_ids) == 2:
        # precompute freqs_cos and freqs_sin
        return position_ids
    else:
        return orig_embedding_fwd(self, x, position_ids)

def replace_qwen2_rms_norm_with_native_rms_norm(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, Qwen2RMSNorm):
            rms_norm = torch.nn.RMSNorm(child.weight.size(0), eps=child.variance_epsilon)
            rms_norm.weight = child.weight
            setattr(
                module,
                name,
                rms_norm,
            )
        else:
            replace_qwen2_rms_norm_with_native_rms_norm(child)
    return module

def initialize_r3_hadamard(module: torch.nn.Module):
    for _, child in module.named_children():
        if isinstance(child, R3Hadamard):
            child.initialize_r3_hadamard()
        else:
            initialize_r3_hadamard(child)
    return module
