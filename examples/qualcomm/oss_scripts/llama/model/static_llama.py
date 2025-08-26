# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import math
from typing import List, Optional, Tuple

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


def apply_rotary_emb_single(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    # The implementation of RoPE in HuggingFace processes query and key with two half instead of interleaved way.
    # The main difference is stride in StrideSlice op. For interleaved way, stride is two which is not friendly for HTP backend.
    # Ref: https://github.com/huggingface/transformers/issues/25199
    x_r, x_i = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # broadcast for batch_prefill mode input x
    if x.dim() == 4:
        freqs_cos = freqs_cos[None, :, None, :]
        freqs_sin = freqs_sin[None, :, None, :]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_out = torch.cat([x_out_r, x_out_i], dim=-1)
    return x_out


def apply_partial_rotary_emb_single(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:

    if x.dim() == 4:
        freqs_cos = freqs_cos[None, :, None, :]
        freqs_sin = freqs_sin[None, :, None, :]

    rotary_dim = freqs_cos.shape[-1] * 2

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_r, x_i = x_rot[..., : x_rot.shape[-1] // 2], x_rot[..., x_rot.shape[-1] // 2 :]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos
    x_rotated = torch.cat([x_out_r, x_out_i], dim=-1)
    return torch.cat([x_rotated, x_pass], dim=-1)


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
        self.output_new_cache_only = output_new_cache_only
        self.enable_masked_softmax = getattr(config, "enable_masked_softmax", False)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_before_rope = config.qk_norm_before_rope
        apply_rope_layers = getattr(config, "apply_rope_layers", None)
        self.use_rope = (
            apply_rope_layers[layer_idx] if apply_rope_layers is not None else True
        )

        if self.use_qk_norm:
            q_norm_dim = self.head_dim
            k_norm_dim = self.head_dim
            self.q_norm_fn = torch.nn.RMSNorm(q_norm_dim, eps=config.norm_eps)
            self.k_norm_fn = torch.nn.RMSNorm(k_norm_dim, eps=config.norm_eps)

        if config.partial_rotary_factor < 1:
            self.apply_rope_emb = apply_partial_rotary_emb_single
        else:
            self.apply_rope_emb = apply_rotary_emb_single

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

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        self.scale = float(self.head_dim) ** 0.5

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

    def prepare_sha(self):
        self.wq_sha = nn.ModuleList(
            [
                nn.Conv2d(
                    self.dim,
                    self.head_dim,
                    1,
                    bias=getattr(self.config, "attention_qkv_bias", False),
                )
                for _ in range(self.n_heads)
            ]
        )
        self.wk_sha = nn.ModuleList(
            [
                nn.Conv2d(
                    self.dim,
                    self.head_dim,
                    1,
                    bias=getattr(self.config, "attention_qkv_bias", False),
                )
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wv_sha = nn.ModuleList(
            [
                nn.Conv2d(
                    self.dim,
                    self.head_dim,
                    1,
                    bias=getattr(self.config, "attention_qkv_bias", False),
                )
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wo_sha = nn.Conv2d(self.n_heads * self.head_dim, self.dim, 1, bias=False)

        self.forward_mha = self.forward
        self.forward = self.forward_sha
        for i in range(self.n_heads):
            self.wq_sha[i].weight.data.copy_(
                self.wq.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            if self.wq_sha[i].bias is not None:
                self.wq_sha[i].bias.data.copy_(
                    self.wq.bias[i * self.head_dim : (i + 1) * self.head_dim]
                )
        for i in range(self.n_kv_heads):
            self.wk_sha[i].weight.data.copy_(
                self.wk.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            if self.wk_sha[i].bias is not None:
                self.wk_sha[i].bias.data.copy_(
                    self.wk.bias[i * self.head_dim : (i + 1) * self.head_dim]
                )
            self.wv_sha[i].weight.data.copy_(
                self.wv.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            if self.wv_sha[i].bias is not None:
                self.wv_sha[i].bias.data.copy_(
                    self.wv.bias[i * self.head_dim : (i + 1) * self.head_dim]
                )
        self.wo_sha.weight.data.copy_(self.wo.weight[:, :, None, None])

    def forward_sha(  # noqa: C901
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: Optional[List[torch.Tensor]] = None,
        v_caches: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        # In the HTP backend, the input axis order for the convolution operation is
        # more efficient with [1, 1, seq_len, dim] compared to [1, seq_len, 1, dim].
        hidden_states = torch.reshape(
            hidden_states, (bsz, seq_len, 1, self.dim)
        ).transpose(1, 3)
        q = [
            wq_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wq_sha in self.wq_sha
        ]
        k = [
            wk_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wk_sha in self.wk_sha
        ]
        v = [
            wv_sha(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(bsz, seq_len, self.head_dim)
            for wv_sha in self.wv_sha
        ]

        for i in range(len(q)):
            if self.use_qk_norm and self.qk_norm_before_rope:
                q[i] = self.q_norm_fn(q[i])
            if self.use_rope:
                q[i] = self.apply_rope_emb(q[i], freqs_cos, freqs_sin)
            if self.use_qk_norm and not self.qk_norm_before_rope:
                q[i] = self.q_norm_fn(q[i])
            if getattr(self.config, "enable_r3", False):
                q[i] = torch.matmul(q[i], self.r3_weight)

        for i in range(len(k)):
            if self.use_qk_norm and self.qk_norm_before_rope:
                k[i] = self.k_norm_fn(k[i])
            if self.use_rope:
                k[i] = self.apply_rope_emb(k[i], freqs_cos, freqs_sin)
            if self.use_qk_norm and not self.qk_norm_before_rope:
                k[i] = self.k_norm_fn(k[i])
            if getattr(self.config, "enable_r3", False):
                k[i] = torch.matmul(k[i], self.r3_weight)
            k[i] = k[i].transpose(1, 2)

        output_y = []
        kh, vh = [], []
        # kv cache mode
        if k_caches and v_caches:
            for i, _ in enumerate(k_caches):
                kh.append(torch.cat([k_caches[i], k[i]], dim=-1))
                vh.append(torch.cat([v_caches[i], v[i]], dim=1))
        # batch_prefill mode
        else:
            kh = k
            vh = v

        for i, _ in enumerate(q):
            cache_idx = i // self.num_key_value_groups
            attn = q[i] @ kh[cache_idx]
            attn = attn / self.scale
            if self.enable_masked_softmax:
                attn_min = torch.amin(attn, dim=-1, keepdim=True)
                minus_value = -20
                attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
            else:
                attn = attn + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ vh[cache_idx]

            output_y.append(y)

        y = torch.concat(output_y, dim=-1)
        y = y.reshape(bsz, seq_len, 1, -1)
        y = y.transpose(1, 3)
        y = self.wo_sha(y)
        y = y.transpose(1, 3)
        y = y.reshape(bsz, seq_len, -1)

        if self.output_new_cache_only:
            return y, k, v

        return y, kh, vh

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

        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        if self.use_rope:
            q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
            k = self.apply_rope_emb(k, freqs_cos, freqs_sin)
        k = k.permute(0, 2, 3, 1)

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        output_kh, output_vh, output_y = [], [], []
        kh, vh = [], []
        # kv cache mode
        if k_caches and v_caches:
            for i, _ in enumerate(k_caches):
                kh.append(torch.cat([k_caches[i], k[:, i, :, :]], dim=-1))
                vh.append(torch.cat([v_caches[i], v[:, :, i, :]], dim=1))
            for i in range(self.n_heads):
                cache_idx = i // self.num_key_value_groups

                attn = q[:, :, i, :] @ kh[cache_idx]
                attn = attn / self.scale + atten_mask
                attn = self.attn_softmax(attn)
                y = attn @ vh[cache_idx]

                output_y.append(y)

        # batch_prefill mode
        else:
            kh = k
            vh = v
            for i in range(self.n_heads):
                cache_idx = i // self.num_key_value_groups

                attn = q[:, :, i, :] @ kh[:, cache_idx, :, :]
                attn = attn / self.scale + atten_mask
                attn = self.attn_softmax(attn)
                y = attn @ vh[:, :, cache_idx, :]

                output_y.append(y)

        for i in range(self.n_kv_heads):
            if self.output_new_cache_only:
                output_kh.append(k[:, i, :, -1])
                output_vh.append(v[:, -1, i, :])
            else:
                output_kh.append(k[:, i, :, :])
                output_vh.append(v[:, :, i, :])

        y = torch.concat(output_y, dim=-1)
        y = self.wo(y)

        return y, output_kh, output_vh


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

    def prepare_feedfoward_conv(self):
        self.w1_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)
        self.w2_conv = nn.Conv2d(self.hidden_dim, self.dim, 1, bias=False)
        self.w3_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_feedfoward_conv

        self.w1_conv.weight.data.copy_(self.w1.weight[:, :, None, None])
        self.w2_conv.weight.data.copy_(self.w2.weight[:, :, None, None])
        self.w3_conv.weight.data.copy_(self.w3.weight[:, :, None, None])

        del self.w1
        del self.w2
        del self.w3

    def forward_feedfoward_conv(self, x):
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
        self.feed_forward = FeedForward(config)
        self.attention_norm = torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
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
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k_cache, v_cache = self.attention(
            hidden_states=self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            atten_mask=atten_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        if self.post_attention_norm:
            h = self.post_attention_norm(h)
        h = x + h
        out = self.feed_forward(self.ffn_norm(h))
        if self.post_ffn_norm:
            out = self.post_ffn_norm(out)
        output = h + out
        return output, k_cache, v_cache


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

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(i, config, self.output_new_cache_only)
                for i in range(config.n_layers)
            ]
        )
        self.norm = torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        if config.use_hf_rope:
            freqs_cos, freqs_sin = hf_precompute_freqs_cis(
                config.head_dim,
                config.max_seq_len,
                config.rope_freq_base,
                config.partial_rotary_factor,
            )
            freqs_cos = freqs_cos[:, : freqs_cos.shape[-1] // 2]
            freqs_sin = freqs_sin[:, : freqs_sin.shape[-1] // 2]
        else:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                config.head_dim,
                config.max_seq_len,
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

        hidden_states = self.embedding_scale_factor * self.tok_embeddings(tokens)
        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind * self.n_kv_heads
                offset_v = self.n_layers * self.n_kv_heads + offset_k
                k_caches = args[offset_k : offset_k + self.n_kv_heads]
                v_caches = args[offset_v : offset_v + self.n_kv_heads]
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

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self, use_kv_cache=True):
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, self.ar_len), dtype=dtype
        )
        atten_mask = AttentionMask(
            CausalAttentionMask(self.max_batch_size, self.ar_len, self.max_seq_len)
        )
        if use_kv_cache:
            pos_ids = torch.zeros((self.max_batch_size, self.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []

            for _ in range(self.n_layers):
                for _ in range(self.n_kv_heads):
                    # transpose first to decrease the runtime efforts
                    k_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.head_dim,
                            self.max_seq_len - self.ar_len,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            self.max_batch_size,
                            self.max_seq_len - self.ar_len,
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
        # TODO: modify this when enabling LLAMA 7B
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_dim": self.dim,
            "get_head_dim": self.head_dim,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_kv_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
            "get_kv_io_bit_width": self.kv_io_bit_width,
        }


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
        )

        for key in ["layer_types", "sliding_window", "rope_local_base_freq"]:
            assert key in kwargs, f"Missing required argument: '{key}' in kwargs"

        # Get attention type for each layer
        self.layer_types = kwargs["layer_types"]
        # Get sliding window size (used in local/global attention)
        self.sliding_window = kwargs["sliding_window"]
        # Get local freq base for sliding attention
        rope_freq_base = kwargs["rope_local_base_freq"]

        local_freqs_cos, local_freqs_sin = hf_precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len,
            rope_freq_base,
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

        hidden_states = self.embedding_scale_factor * self.tok_embeddings(tokens)
        for ind, decoder_layer in enumerate(self.layers):
            k_caches = None
            v_caches = None
            if self.use_kv_cache:
                offset_k = ind * self.n_kv_heads
                offset_v = self.n_layers * self.n_kv_heads + offset_k
                k_caches = args[offset_k : offset_k + self.n_kv_heads]
                v_caches = args[offset_v : offset_v + self.n_kv_heads]

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
        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self, use_kv_cache=True):
        inputs = list(super().get_example_inputs(use_kv_cache=use_kv_cache))
        causal_mask = CausalAttentionMask(
            self.max_batch_size, self.ar_len, self.max_seq_len
        )
        sliding_window_mask = SlidingWindowAttentionMask(
            self.max_batch_size,
            self.ar_len,
            self.max_seq_len,
            sliding_window=self.sliding_window,
        )
        # Don't reverse the order of attention mask
        inputs[1] = AttentionMask([causal_mask, sliding_window_mask])
        return tuple(inputs)

    def get_metadata(self):
        meta_data = super().get_metadata()
        meta_data["get_sliding_window"] = self.sliding_window
        return meta_data
