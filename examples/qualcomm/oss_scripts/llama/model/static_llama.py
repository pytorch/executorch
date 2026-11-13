# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import math
from typing import Dict, List, Optional, Tuple, Type

import scipy
import torch
import torch.nn as nn

from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import (
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
    SlidingWindowAttentionMask,
)
from executorch.examples.qualcomm.oss_scripts.llama.model import (
    FeedForward_REGISTRY,
    NORM_REGISTRY,
    ROTARY_EMB_REGISTRY,
)


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


class AttentionSinkRope(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        sink_size: int,
        eviction_batch_size: int,
        ar_len: int,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.sink_size = sink_size
        self.eviction_batch_size = eviction_batch_size
        self.n_layers = config.n_layers
        self.original_position = eviction_batch_size + sink_size
        self.new_position = sink_size
        self.num_to_keep = (
            config.max_context_len - sink_size - eviction_batch_size - ar_len
        )
        self.evict_k_cache_shape = [
            config.max_batch_size,
            config.n_kv_heads,
            config.head_dim,
            eviction_batch_size,
        ]
        self.evict_v_cache_shape = [
            config.max_batch_size,
            config.n_kv_heads,
            eviction_batch_size,
            config.head_dim,
        ]
        self.kv_cache_shape = {
            # single head, k input
            "k": (
                config.max_batch_size,
                config.n_kv_heads,
                config.head_dim,
                config.max_context_len - ar_len,
            ),
            # single head, v input
            "v": (
                config.max_batch_size,
                config.n_kv_heads,
                config.max_context_len - ar_len,
                config.head_dim,
            ),
        }

        if getattr(config, "enable_r3", False):
            self.register_buffer(
                "r3_weight",
                torch.tensor(
                    scipy.linalg.hadamard(config.head_dim, dtype=float)
                    / math.sqrt(config.head_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                persistent=False,
            )

        if config.partial_rotary_factor < 1:
            self.apply_rope_emb = ROTARY_EMB_REGISTRY["partial"]
        else:
            self.apply_rope_emb = ROTARY_EMB_REGISTRY["default"]

        if config.use_hf_rope:
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.partial_rotary_factor,
            )
            freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2]
            freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2]
        else:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.use_scaled_rope,
                config.rope_scale_factor,
            )
        original_freqs_cos = freqs_cos.narrow(
            0, self.original_position, self.num_to_keep
        )
        original_freqs_sin = freqs_sin.narrow(
            0, self.original_position, self.num_to_keep
        )
        new_freqs_cos = freqs_cos.narrow(0, self.new_position, self.num_to_keep)
        new_freqs_sin = freqs_sin.narrow(0, self.new_position, self.num_to_keep)
        rerotation_cos = (
            new_freqs_cos * original_freqs_cos + new_freqs_sin * original_freqs_sin
        )
        rerotation_sin = (
            new_freqs_sin * original_freqs_cos - new_freqs_cos * original_freqs_sin
        )
        self.register_buffer("rerotation_cos", rerotation_cos, persistent=False)
        self.register_buffer("rerotation_sin", rerotation_sin, persistent=False)

        self.sliding_window = kwargs.get("sliding_window", False)
        if self.sliding_window:
            # Get attention type for each layer
            self.layer_types = kwargs["layer_types"]
            # Get local freq base for sliding attention
            rope_freq_base = kwargs["rope_local_base_freq"]
            local_freqs_cos, local_freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                rope_freq_base,
                config.partial_rotary_factor,
            )
            local_freqs_cos = local_freqs_cos[:, : local_freqs_cos.shape[-1] // 2]
            local_freqs_sin = local_freqs_sin[:, : local_freqs_sin.shape[-1] // 2]
            local_original_freqs_cos = local_freqs_cos.narrow(
                0, self.original_position, self.num_to_keep
            )
            local_original_freqs_sin = local_freqs_sin.narrow(
                0, self.original_position, self.num_to_keep
            )
            local_new_freqs_cos = local_freqs_cos.narrow(
                0, self.new_position, self.num_to_keep
            )
            local_new_freqs_sin = local_freqs_sin.narrow(
                0, self.new_position, self.num_to_keep
            )
            local_rerotation_cos = (
                local_new_freqs_cos * local_original_freqs_cos
                + local_new_freqs_sin * local_original_freqs_sin
            )
            local_rerotation_sin = (
                local_new_freqs_sin * local_original_freqs_cos
                - local_new_freqs_cos * local_original_freqs_sin
            )
            self.register_buffer(
                "local_rerotation_cos", local_rerotation_cos, persistent=False
            )
            self.register_buffer(
                "local_rerotation_sin", local_rerotation_sin, persistent=False
            )

    def forward(self, k_caches: List[torch.Tensor], v_caches: List[torch.Tensor]):
        """
        Rerotate k_cache from original_position to new_position, and return the kv cache after eviction. This is done by rerotating
        k_cache with (new_position * theta - original_position * theta) with the following matrix:
        (cos(delta), -sin(delta)
         sin(delta), cos(delta))
         where delta = new_position * theta - original_position * theta

         Based on https://github.com/huggingface/transformers/pull/26681
        """

        output_k_caches, output_v_caches = [], []
        for ind, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
            # k_cache: (batch_size, n_kv_heads, head_dim, seq_len)
            # v_cache: (batch_size, n_kv_heads, seq_len, head_dim)
            k_dim_to_slice = 3
            v_dim_to_slice = 2

            k_to_keep = k_cache.narrow(
                k_dim_to_slice,
                self.original_position,
                self.num_to_keep,
            )
            k_to_keep = k_to_keep.transpose(2, 3)
            if getattr(self.config, "enable_r3", False):
                # We need to revert the key from spin quant before applying RoPE
                k_to_keep = torch.matmul(k_to_keep, self.r3_weight.T)

            if self.sliding_window and self.layer_types[ind] == "sliding_attention":
                k_to_keep = self.apply_rope_emb(
                    k_to_keep, self.local_rerotation_cos, self.local_rerotation_sin
                )
            else:
                k_to_keep = self.apply_rope_emb(
                    k_to_keep, self.rerotation_cos, self.rerotation_sin
                )
            if getattr(self.config, "enable_r3", False):
                k_to_keep = torch.matmul(k_to_keep, self.r3_weight)
            k_to_keep = k_to_keep.transpose(2, 3)
            new_k_cache = torch.cat(
                [
                    k_cache.narrow(k_dim_to_slice, 0, self.sink_size),
                    k_to_keep,
                    torch.zeros(self.evict_k_cache_shape),
                ],
                dim=k_dim_to_slice,
            )

            new_v_cache = torch.cat(
                [
                    v_cache.narrow(v_dim_to_slice, 0, self.sink_size),
                    v_cache.narrow(
                        v_dim_to_slice,
                        self.original_position,
                        self.num_to_keep,
                    ),
                    torch.zeros(self.evict_v_cache_shape),
                ],
                dim=v_dim_to_slice,
            )

            output_k_caches.append(new_k_cache)
            output_v_caches.append(new_v_cache)

        return output_k_caches, output_v_caches

    def get_example_inputs(self):
        k_cache, v_cache = [], []

        for _ in range(self.n_layers):
            k_cache.append(torch.zeros(self.kv_cache_shape["k"]))
            v_cache.append(torch.zeros(self.kv_cache_shape["v"]))
        return k_cache, v_cache

    def get_metadata(self):
        return {
            "get_eviction_batch_size": self.eviction_batch_size,
            "get_max_context_len": self.config.max_context_len,
            "get_sink_size": self.sink_size,
        }


class LlamaAttention(nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = config.n_heads // self.n_kv_heads
        self.max_seq_len = config.max_seq_len
        self.max_context_len = config.max_context_len
        self.use_kv_cache = config.use_kv_cache
        self.output_new_cache_only = output_new_cache_only
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_before_rope = config.qk_norm_before_rope
        # If None, assume each layer uses rope
        self.use_rope = (
            config.no_rope_layer_interval is None
            or (layer_idx + 1) % config.no_rope_layer_interval
        )

        if self.use_qk_norm:
            q_norm_dim = self.head_dim
            k_norm_dim = self.head_dim
            self.q_norm_fn = torch.nn.RMSNorm(q_norm_dim, eps=config.norm_eps)
            self.k_norm_fn = torch.nn.RMSNorm(k_norm_dim, eps=config.norm_eps)

        partial_rotary_factor = config.partial_rotary_factor
        if config.rope_parameters and config.layer_types:
            partial_rotary_factor = config.rope_parameters[
                config.layer_types[layer_idx]
            ].get("partial_rotary_factor", 1.0)
        if partial_rotary_factor < 1:
            self.apply_rope_emb = ROTARY_EMB_REGISTRY["partial"]
        else:
            self.apply_rope_emb = ROTARY_EMB_REGISTRY["default"]

        self.wq = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=getattr(config, "attention_qkv_bias", False),
        )
        self.wk = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=getattr(config, "attention_qkv_bias", False),
        )
        self.wv = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=getattr(config, "attention_qkv_bias", False),
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        if self.headwise_attn_output_gate:
            self.og = nn.Linear(self.dim, self.n_heads, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        self.scale = (
            float(self.head_dim) ** 0.5
            if config.attention_multiplier is None
            else 1.0 / config.attention_multiplier
        )

        # gemma 2 uses soft-capping on attention and logits
        self.attn_logit_softcapping = config.attn_logit_softcapping

        if getattr(config, "enable_r3", False):
            self.register_buffer(
                "r3_weight",
                torch.tensor(
                    scipy.linalg.hadamard(self.head_dim, dtype=float)
                    / math.sqrt(self.head_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                persistent=False,
            )

    def prepare_attention_conv(self):
        self.wq_conv = nn.Conv2d(
            self.dim,
            self.n_heads * self.head_dim,
            1,
            bias=getattr(self.config, "attention_qkv_bias", False),
        )
        self.wk_conv = nn.Conv2d(
            self.dim,
            self.n_kv_heads * self.head_dim,
            1,
            bias=getattr(self.config, "attention_qkv_bias", False),
        )
        self.wv_conv = nn.Conv2d(
            self.dim,
            self.n_kv_heads * self.head_dim,
            1,
            bias=getattr(self.config, "attention_qkv_bias", False),
        )
        self.wo_conv = nn.Conv2d(self.n_heads * self.head_dim, self.dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_attention_conv

        self.wq_conv.weight.data.copy_(self.wq.weight[:, :, None, None])
        if self.wq_conv.bias is not None:
            self.wq_conv.bias.data.copy_(self.wq.bias)
        self.wk_conv.weight.data.copy_(self.wk.weight[:, :, None, None])
        if self.wk_conv.bias is not None:
            self.wk_conv.bias.data.copy_(self.wk.bias)
        self.wv_conv.weight.data.copy_(self.wv.weight[:, :, None, None])
        if self.wv_conv.bias is not None:
            self.wv_conv.bias.data.copy_(self.wv.bias)
        self.wo_conv.weight.data.copy_(self.wo.weight[:, :, None, None])

        del self.wq
        del self.wk
        del self.wv
        del self.wo

    def forward_attention_conv(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        if self.headwise_attn_output_gate:
            gate = torch.sigmoid(self.og(hidden_states)).unsqueeze(-1)
        hidden_states = torch.reshape(
            hidden_states, (bsz, seq_len, 1, self.dim)
        ).transpose(1, 3)

        q = self.wq_conv(hidden_states)
        k = self.wk_conv(hidden_states)
        v = self.wv_conv(hidden_states)
        q = q.permute(0, 3, 1, 2).squeeze(-1)
        k = k.permute(0, 3, 1, 2).squeeze(-1)
        v = v.permute(0, 3, 1, 2).squeeze(-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_rope:
            q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
            k = self.apply_rope_emb(k, freqs_cos, freqs_sin)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        kh, vh = None, None
        # kv cache mode
        if self.use_kv_cache:
            kh = torch.cat([k_caches, k], dim=-1)
            vh = torch.cat([v_caches, v], dim=2)
        # batch_prefill mode
        else:
            kh = k
            vh = v

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2)
        if self.headwise_attn_output_gate:
            y = y * gate
        y = y.reshape(bsz, seq_len, 1, -1).transpose(1, 3)
        y = self.wo_conv(y)
        y = y.transpose(1, 3)
        y = y.reshape(bsz, seq_len, -1)

        if self.output_new_cache_only:
            return y, [k], [v]

        return y, [kh], [vh]

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        if self.headwise_attn_output_gate:
            gate = torch.sigmoid(self.og(hidden_states)).unsqueeze(-1)

        q = self.wq(hidden_states)
        k = self.wk(hidden_states)
        v = self.wv(hidden_states)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_rope:
            q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
            k = self.apply_rope_emb(k, freqs_cos, freqs_sin)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)
        if getattr(self.config, "enable_r3", False):
            q = torch.matmul(q, self.r3_weight)
            k = torch.matmul(k, self.r3_weight)
        k = k.transpose(2, 3)

        kh, vh = None, None
        # kv cache mode
        if self.use_kv_cache:
            kh = torch.cat([k_caches, k], dim=-1)
            vh = torch.cat([v_caches, v], dim=2)
        # batch_prefill mode
        else:
            kh = k
            vh = v

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        # gemma2-2b
        if self.attn_logit_softcapping is not None:
            attn = attn / self.attn_logit_softcapping
            attn = torch.tanh(attn)
            attn = attn * self.attn_logit_softcapping
        attn = attn / self.scale
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh
        y = y.transpose(1, 2)
        if self.headwise_attn_output_gate:
            y = y * gate
        y = y.reshape(bsz, seq_len, -1)
        y = self.wo(y)

        if self.output_new_cache_only:
            return y, [k], [v]

        return y, [kh], [vh]


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.hidden_dim is not None
        self.hidden_dim: int = args.hidden_dim
        self.dim: int = args.dim
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.act_fn = args.act_fn.get_function()

    def prepare_feedforward_conv(self):
        self.w1_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)
        self.w2_conv = nn.Conv2d(self.hidden_dim, self.dim, 1, bias=False)
        self.w3_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_feedforward_conv
        self.w1_conv.weight.data.copy_(self.w1.weight[:, :, None, None])
        self.w2_conv.weight.data.copy_(self.w2.weight[:, :, None, None])
        self.w3_conv.weight.data.copy_(self.w3.weight[:, :, None, None])

        del self.w1
        del self.w2
        del self.w3

    def forward_feedforward_conv(self, x):
        bsz, _, _ = x.size()
        x = torch.reshape(x, (bsz, -1, 1, self.dim))
        x = x.transpose(1, 3)  # Transpose right before and after Conv
        x = self.w2_conv(self.act_fn(self.w1_conv(x)) * self.w3_conv(x))
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, -1, self.dim))
        return x

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = config.dim
        self.attention = LlamaAttention(
            layer_idx=layer_idx,
            config=config,
            output_new_cache_only=output_new_cache_only,
        )

        self.feed_forward = FeedForward_REGISTRY.get(
            config.model_architecture, FeedForward
        )(config)
        self.attention_norm = NORM_REGISTRY[config.norm_type](
            config.dim, eps=config.norm_eps
        )
        self.ffn_norm = (
            NORM_REGISTRY[config.norm_type](config.dim, eps=config.norm_eps)
            if config.use_ffn_norm
            else None
        )

        self.post_attention_norm = (
            torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
            if config.post_attention_norm
            else None
        )
        self.residual_multiplier = config.residual_multiplier
        self.post_ffn_norm = (
            torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
            if config.post_ffn_norm
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        hidden_states = self.attention_norm(x)
        h, k_cache, v_cache = self.attention(
            hidden_states=hidden_states,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            atten_mask=atten_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        if self.post_attention_norm:
            h = self.post_attention_norm(h)
        h = (
            x + h * self.residual_multiplier
            if self.residual_multiplier is not None
            else x + h
        )
        hidden_states = hidden_states if self.ffn_norm is None else self.ffn_norm(h)
        out = self.feed_forward(hidden_states)
        if self.post_ffn_norm:
            out = self.post_ffn_norm(out)
        output = (
            h + out * self.residual_multiplier
            if self.residual_multiplier is not None
            else h + out
        )

        return output, k_cache, v_cache


DECODER_LAYER_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_decoder_layer(layer_type: str):
    """Register a decoder layer class for a `layer_types` entry."""

    def decorator(cls: Type[nn.Module]):
        DECODER_LAYER_REGISTRY[layer_type] = cls
        return cls

    return decorator


class ShortConv(nn.Module):
    """LFM2/LFM3 gated depthwise causal conv over the last `conv_L_cache` inputs."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.L_cache = config.conv_L_cache
        self.B_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.C_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.x_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        # Only holds the depthwise weight [dim, 1, L_cache]; each tap is applied as a
        # broadcast multiply over [bsz, seq, dim].
        self.conv = nn.Conv1d(
            self.dim, self.dim, self.L_cache, groups=self.dim, bias=False
        )

    def forward(
        self, x: torch.Tensor, conv_state: torch.Tensor, window_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        conv_in = self.B_proj(x) * self.x_proj(x)

        # window_mask zeroes state rows that precede position 0, so a fresh
        # sequence needs no knowledge of the state's quantized zero point.
        window = torch.cat([conv_state, conv_in], dim=1) * window_mask
        weight = self.conv.weight
        y = window[:, 0:seq_len, :] * weight[:, 0, 0]
        for k in range(1, self.L_cache):
            y = y + window[:, k : k + seq_len, :] * weight[:, 0, k]

        return self.out_proj(self.C_proj(x) * y), conv_in


@register_decoder_layer("conv")
class ShortConvDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.conv = ShortConv(config)
        self.feed_forward = FeedForward_REGISTRY.get(
            config.model_architecture, FeedForward
        )(config)
        self.attention_norm = NORM_REGISTRY[config.norm_type](
            config.dim, eps=config.norm_eps
        )
        self.ffn_norm = NORM_REGISTRY[config.norm_type](config.dim, eps=config.norm_eps)
        self.post_attention_norm = (
            torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
            if config.post_attention_norm
            else None
        )
        self.post_ffn_norm = (
            torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
            if config.post_ffn_norm
            else None
        )

    def forward(
        self, x: torch.Tensor, conv_state: torch.Tensor, window_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, conv_in = self.conv(self.attention_norm(x), conv_state, window_mask)
        if self.post_attention_norm:
            h = self.post_attention_norm(h)
        h = x + h
        out = self.feed_forward(self.ffn_norm(h))
        if self.post_ffn_norm:
            out = self.post_ffn_norm(out)
        return h + out, conv_in


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.max_context_len = config.max_context_len
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.rope_freq_base = config.rope_freq_base
        self.use_kv_cache = config.use_kv_cache
        self.embedding_scale_factor = config.embedding_scale_factor
        self.ar_len = ar_len
        self.output_new_cache_only = output_new_cache_only
        self.use_i64_token = use_i64_token
        self.output_cache = output_cache
        self.kv_io_bit_width = config.kv_io_bit_width
        self.logits_scaling = config.logits_scaling

        layer_types = config.layer_types or [None] * config.n_layers
        self.layers = nn.ModuleList(
            [
                DECODER_LAYER_REGISTRY.get(layer_types[i], LlamaDecoderLayer)(
                    i, config, self.output_new_cache_only
                )
                for i in range(config.n_layers)
            ]
        )
        self.norm = NORM_REGISTRY[config.norm_type](config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=config.output_bias)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        if config.use_hf_rope:
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.partial_rotary_factor,
            )
            freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2]
            freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2]
        else:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                config.head_dim,
                config.max_context_len,
                config.rope_freq_base,
                config.use_scaled_rope,
                config.rope_scale_factor,
            )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def prepare_output_conv(self):
        def forward_output_conv(x):
            bsz, _, _ = x.size()
            x = torch.reshape(x, (bsz, -1, 1, self.dim))
            x = x.transpose(1, 3)  # Transpose right before and after Conv
            x = self.output_conv(x)
            x = x.transpose(1, 3)
            x = torch.reshape(x, (bsz, -1, self.vocab_size))
            return x

        self.output_conv = nn.Conv2d(self.dim, self.vocab_size, 1, bias=False)
        self.output_conv.weight.data.copy_(self.output.weight[:, :, None, None])

        del self.output
        self.output = forward_output_conv

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        hidden_states = self.tok_embeddings(tokens)

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.logits_scaling:
            logits = logits / self.logits_scaling

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )
        atten_mask = AttentionMask(
            CausalAttentionMask(self.max_batch_size, self.ar_len, self.max_context_len)
        )
        if self.use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []
            for _ in range(self.n_layers):
                # transpose first to decrease the runtime efforts
                k_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.head_dim,
                        self.max_context_len - self.ar_len,
                    )
                )
                v_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.max_context_len - self.ar_len,
                        self.head_dim,
                    )
                )
            return (
                tokens,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            tokens,
            atten_mask,
        )

    def get_metadata(self):
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_dim": self.dim,
            "get_head_dim": self.head_dim,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_max_context_len": self.max_context_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_kv_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
            "get_kv_io_bit_width": self.kv_io_bit_width,
        }


class LlamaModelWithoutEmbedding(LlamaModel):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
            **kwargs,
        )

        # Set the audio/image token ID from keyword arguments. It defaults to None if not provided.
        # If an ID is provided, it will be stored in the model's metadata.
        self.audio_token_id = kwargs.get("audio_token_id", None)
        self.image_token_id = kwargs.get("image_token_id", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )

        hidden_states = (
            self.embedding_scale_factor * hidden_states
            if self.embedding_scale_factor != 1.0
            else hidden_states
        )

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                atten_mask=atten_mask,
                k_caches=k_caches,
                v_caches=v_caches,
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.logits_scaling:
            logits = logits / self.logits_scaling

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        hidden_states = torch.randn(
            (self.max_batch_size, self.ar_len, self.dim), dtype=torch.float32
        )
        atten_mask = AttentionMask(
            CausalAttentionMask(self.max_batch_size, self.ar_len, self.max_seq_len)
        )
        if self.use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []

            for _ in range(self.n_layers):
                # transpose first to decrease the runtime efforts
                k_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.head_dim,
                        self.max_seq_len - self.ar_len,
                    )
                )
                v_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.n_kv_heads,
                        self.max_seq_len - self.ar_len,
                        self.head_dim,
                    )
                )
            return (
                hidden_states,
                atten_mask,
                pos_ids,
                k_cache,
                v_cache,
            )

        return (
            hidden_states,
            atten_mask,
        )

    def get_metadata(self):
        meta_data = super().get_metadata()
        if self.audio_token_id:
            meta_data["audio_token_id"] = self.audio_token_id
        if self.image_token_id:
            meta_data["image_token_id"] = self.image_token_id
        return meta_data


class MultiScopeAwareLlamaModel(LlamaModel):
    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
            **kwargs,
        )
        # Parameter final_logit_softcapping is not necessary for all
        self.final_logit_softcapping = config.final_logit_softcapping

        # Gemma2/Gemma3 requires additional configuration parameters:
        # - layer_types: Specifies the type of each layer (e.g., full vs. sliding attention)
        # - local_rope_theta: Base frequency for local RoPE
        # - sliding_window: Size of the sliding window for local attention
        self.layer_types = config.layer_types
        if self.layer_types is not None:
            assert len(self.layer_types) == self.n_layers, (
                f"Length of layer_types ({len(self.layer_types)}) must match "
                f"n_layers ({self.n_layers})"
            )
        assert (
            config.local_rope_theta is not None
        ), "local_rope_theta should not be None, please set it explicitly in config."

        self.sliding_window = config.sliding_window
        local_freqs_cos, local_freqs_sin = hf_precompute_freqs_cis(
            config.head_dim,
            config.max_context_len,
            config.local_rope_theta,
            config.partial_rotary_factor,
        )
        local_freqs_cos = local_freqs_cos[:, : local_freqs_cos.shape[-1] // 2]
        local_freqs_sin = local_freqs_sin[:, : local_freqs_sin.shape[-1] // 2]
        self.register_buffer("local_freqs_cos", local_freqs_cos, persistent=False)
        self.register_buffer("local_freqs_sin", local_freqs_sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = (
            self.freqs_cos[input_pos][0] if self.use_kv_cache else self.freqs_cos
        )
        freqs_sin = (
            self.freqs_sin[input_pos][0] if self.use_kv_cache else self.freqs_sin
        )
        local_freqs_cos = (
            self.local_freqs_cos[input_pos][0]
            if self.use_kv_cache
            else self.local_freqs_cos
        )
        local_freqs_sin = (
            self.local_freqs_sin[input_pos][0]
            if self.use_kv_cache
            else self.local_freqs_sin
        )

        hidden_states = self.tok_embeddings(tokens)

        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind
                offset_v = self.n_layers + offset_k
                k_caches = args[offset_k]
                v_caches = args[offset_v]

            if self.layer_types[ind] == "sliding_attention":
                hidden_states, k, v = decoder_layer(
                    hidden_states,
                    freqs_cos=local_freqs_cos,
                    freqs_sin=local_freqs_sin,
                    atten_mask=window_atten_mask,
                    k_caches=k_caches,
                    v_caches=v_caches,
                )
            else:
                hidden_states, k, v = decoder_layer(
                    hidden_states,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    atten_mask=atten_mask,
                    k_caches=k_caches,
                    v_caches=v_caches,
                )

            output_k_cache.extend(k)
            output_v_cache.extend(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)
        if self.final_logit_softcapping:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self):
        inputs = list(super().get_example_inputs())
        causal_mask = CausalAttentionMask(
            self.max_batch_size, self.ar_len, self.max_context_len
        )
        sliding_window_mask = SlidingWindowAttentionMask(
            self.max_batch_size,
            self.ar_len,
            self.max_context_len,
            sliding_window=self.sliding_window,
        )
        # Don't reverse the order of attention mask
        inputs[1] = AttentionMask([causal_mask, sliding_window_mask])
        return tuple(inputs)

    def get_metadata(self):
        meta_data = super().get_metadata()
        meta_data["get_sliding_window"] = self.sliding_window
        return meta_data


class ShortConvHybridLlamaModel(LlamaModel):
    """Decoder mixing short-conv layers with full and sliding-window attention.

    forward(tokens, atten_mask, window_atten_mask, input_pos,
            *k_caches[n_attn], *v_caches[n_attn], *conv_states[n_conv])
        -> (logits, new_k[n_attn], new_v[n_attn] + new_conv_inputs[n_conv])

    Conv layers return the conv inputs of the current tokens; the runner keeps the
    last `conv_L_cache - 1` valid rows as the next conv state. They ride along with
    the V caches so the decoder plumbing (calibration, export flattening) sees the
    usual (tokens, masks, pos, k_caches, v_caches) signature.
    """

    def __init__(
        self,
        config: ModelArgs,
        ar_len=1,
        output_new_cache_only=True,
        output_cache=True,
        use_i64_token=False,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ar_len=ar_len,
            output_new_cache_only=output_new_cache_only,
            output_cache=output_cache,
            use_i64_token=use_i64_token,
            **kwargs,
        )
        if not output_new_cache_only or not self.use_kv_cache:
            raise ValueError(
                "LFM3 requires KV caching and new-cache outputs; "
                "prefill_ar_len must be smaller than max_context_len"
            )
        self.layer_types = config.layer_types
        self.sliding_window = config.sliding_window
        self.attention_layers = [
            i for i, t in enumerate(self.layer_types) if t != "conv"
        ]
        self.conv_layers = [i for i, t in enumerate(self.layer_types) if t == "conv"]
        self.conv_state_len = config.conv_L_cache - 1

        # Row r of a conv state holds position input_pos[0] - conv_state_len + r.
        # Kept independent of ar_len: hybrid export copies buffers from the
        # calibration graph into the prefill/decode graphs.
        offsets = torch.arange(-self.conv_state_len, 0)
        positions = torch.arange(self.max_context_len)[:, None] + offsets[None, :]
        self.register_buffer(
            "conv_state_mask",
            (positions >= 0).float().unsqueeze(-1),
            persistent=False,
        )

        # Full-attention theta in the shared config is HF theta ** partial_rotary_factor.
        for prefix, layer_type in (
            ("", "full_attention"),
            ("local_", "sliding_attention"),
        ):
            rope = config.rope_parameters[layer_type]
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                self.max_context_len,
                rope["rope_theta"],
                rope.get("partial_rotary_factor", 1.0),
            )
            self.register_buffer(
                f"{prefix}freqs_cos",
                freqs_cos[:, : freqs_cos.shape[-1] // 2],
                persistent=False,
            )
            self.register_buffer(
                f"{prefix}freqs_sin",
                freqs_sin[:, : freqs_sin.shape[-1] // 2],
                persistent=False,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        n_attn = len(self.attention_layers)
        k_caches = args[:n_attn]
        v_caches = args[n_attn : 2 * n_attn]
        conv_states = args[2 * n_attn :]

        freqs_cos = self.freqs_cos[input_pos][0]
        freqs_sin = self.freqs_sin[input_pos][0]
        local_freqs_cos = self.local_freqs_cos[input_pos][0]
        local_freqs_sin = self.local_freqs_sin[input_pos][0]

        hidden_states = self.tok_embeddings(tokens)

        # select, not slice: a [:, :1] slice is a no-op alias when ar_len == 1,
        # which breaks the graph matching used to share encodings across graphs.
        conv_window_mask = torch.cat(
            [
                self.conv_state_mask[input_pos[:, 0]],
                torch.ones_like(hidden_states[:, :, 0:1]),
            ],
            dim=1,
        )
        output_k_cache, output_v_cache, output_conv = [], [], []
        attn_idx, conv_idx = 0, 0
        for layer_type, decoder_layer in zip(self.layer_types, self.layers):
            if layer_type == "conv":
                hidden_states, conv_in = decoder_layer(
                    hidden_states, conv_states[conv_idx], conv_window_mask
                )
                output_conv.append(conv_in)
                conv_idx += 1
                continue
            if layer_type == "sliding_attention":
                cos, sin, mask = local_freqs_cos, local_freqs_sin, window_atten_mask
            else:
                cos, sin, mask = freqs_cos, freqs_sin, atten_mask
            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=cos,
                freqs_sin=sin,
                atten_mask=mask,
                k_caches=k_caches[attn_idx],
                v_caches=v_caches[attn_idx],
            )
            output_k_cache.extend(k)
            output_v_cache.extend(v)
            attn_idx += 1

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        if self.output_cache:
            return logits, output_k_cache, output_v_cache + output_conv
        return logits

    def get_example_inputs(self):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )
        atten_mask = AttentionMask(
            [
                CausalAttentionMask(
                    self.max_batch_size, self.ar_len, self.max_context_len
                ),
                SlidingWindowAttentionMask(
                    self.max_batch_size,
                    self.ar_len,
                    self.max_context_len,
                    sliding_window=self.sliding_window,
                ),
            ]
        )
        pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
        cache_len = self.max_context_len - self.ar_len
        k_caches = [
            torch.zeros(self.max_batch_size, self.n_kv_heads, self.head_dim, cache_len)
            for _ in self.attention_layers
        ]
        v_caches = [
            torch.zeros(self.max_batch_size, self.n_kv_heads, cache_len, self.head_dim)
            for _ in self.attention_layers
        ]
        conv_states = [
            torch.zeros(self.max_batch_size, self.conv_state_len, self.dim)
            for _ in self.conv_layers
        ]
        return (tokens, atten_mask, pos_ids, k_caches, v_caches + conv_states)

    def get_metadata(self):
        return {
            **super().get_metadata(),
            "get_n_self_layers": len(self.attention_layers),
            "get_n_conv_layers": len(self.conv_layers),
            "get_conv_state_len": self.conv_state_len,
            "get_sliding_window": self.sliding_window,
        }
