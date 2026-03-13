# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.examples.models.llama.rope import precompute_freqs_cis
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    apply_rotary_emb_single,
)
from transformers.configuration_utils import PretrainedConfig


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def prepare_torch_rms_norm(self):
        self.rms_norm = torch.nn.RMSNorm(self.hidden_size, eps=self.variance_epsilon)
        self.rms_norm.weight = self.weight
        self.forward = self.forward_torch

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def forward_torch(self, hidden_states):
        return self.rms_norm(hidden_states)


class Qwen2Attention(nn.Module):
    def __init__(self, config: PretrainedConfig, output_new_cache_only=False):
        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = self.dim // self.n_heads
        self.n_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.max_seq_len = config.max_position_embeddings
        self.output_new_cache_only = output_new_cache_only

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        self.scale = float(self.head_dim) ** 0.5

    def prepare_sha(self):
        self.wq_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=True)
                for _ in range(self.n_heads)
            ]
        )
        self.wk_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=True)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wv_sha = nn.ModuleList(
            [
                nn.Conv2d(self.dim, self.head_dim, 1, bias=True)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wo_sha = nn.Conv2d(self.n_heads * self.head_dim, self.dim, 1, bias=False)

        self.forward_mha = self.forward
        self.forward = self.forward_sha
        for i in range(self.n_heads):
            self.wq_sha[i].weight.data.copy_(
                self.q_proj.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            self.wq_sha[i].bias.data.copy_(
               self.q_proj.bias[i * self.head_dim : (i + 1) * self.head_dim]
            )
        for i in range(self.n_kv_heads):
            self.wk_sha[i].weight.data.copy_(
                self.k_proj.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            self.wk_sha[i].bias.data.copy_(
               self.k_proj.bias[i * self.head_dim : (i + 1) * self.head_dim]
            )
            self.wv_sha[i].weight.data.copy_(
                self.v_proj.weight[
                    i * self.head_dim : (i + 1) * self.head_dim, :, None, None
                ]
            )
            self.wv_sha[i].bias.data.copy_(
               self.v_proj.bias[i * self.head_dim : (i + 1) * self.head_dim]
            )
        self.wo_sha.weight.data.copy_(self.o_proj.weight[:, :, None, None])

    def forward_sha(
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
            q[i] = apply_rotary_emb_single(q[i], freqs_cos, freqs_sin)
        for i in range(len(k)):
            k[i] = apply_rotary_emb_single(k[i], freqs_cos, freqs_sin).transpose(1, 2)

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
            attn = attn / self.scale + atten_mask
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

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb_single(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb_single(k, freqs_cos, freqs_sin).permute(0, 2, 3, 1)

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
        y = self.o_proj(y)

        return y, output_kh, output_vh


class Qwen2MLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_dim: int = config.intermediate_size
        self.dim: int = config.hidden_size
        self.gate_proj = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.up_proj = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def prepare_feedfoward_conv(self):
        self.w1_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)
        self.w2_conv = nn.Conv2d(self.hidden_dim, self.dim, 1, bias=False)
        self.w3_conv = nn.Conv2d(self.dim, self.hidden_dim, 1, bias=False)

        self.forward_no_conv = self.forward
        self.forward = self.forward_feedfoward_conv

        self.w1_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None])
        self.w2_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None])
        self.w3_conv.weight.data.copy_(self.up_proj.weight[:, :, None, None])

        del self.gate_proj
        del self.down_proj
        del self.up_proj

    def forward_feedfoward_conv(self, x):
        bsz, _, _ = x.size()
        x = torch.reshape(x, (bsz, -1, 1, self.dim))
        x = x.transpose(1, 3)  # Transpose right before and after Conv
        x = self.w2_conv(F.silu(self.w1_conv(x)) * self.w3_conv(x))
        x = x.transpose(1, 3)
        x = torch.reshape(x, (bsz, -1, self.dim))
        return x

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, output_new_cache_only=False):
        super().__init__()
        self.dim = config.hidden_size
        self.self_attn = Qwen2Attention(
            config=config, output_new_cache_only=output_new_cache_only
        )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k_cache, v_cache = self.self_attn(
            hidden_states=self.input_layernorm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            atten_mask=atten_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        h = x + h
        output = h + self.mlp(self.post_attention_layernorm(h))
        return output, k_cache, v_cache


class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        ar_len=1,
        output_new_cache_only=True,
        use_i64_token=False,
        max_batch_size=1,
    ):
        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = self.dim // self.n_heads
        self.max_batch_size = max_batch_size
        self.max_seq_len = config.max_position_embeddings
        self.n_kv_heads = config.num_key_value_heads
        self.n_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.rope_freq_base = config.rope_theta
        self.use_kv_cache = config.use_cache
        self.ar_len = ar_len
        self.output_new_cache_only = output_new_cache_only
        self.use_i64_token = use_i64_token

        self.embed_tokens = nn.Embedding(self.vocab_size, self.dim)
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, self.output_new_cache_only)
                for _ in range(self.n_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(self.dim, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.head_dim,
            self.max_seq_len,
            self.rope_freq_base,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

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

        hidden_states = self.embed_tokens(tokens)
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
        return hidden_states, output_k_cache, output_v_cache


class Qwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        ar_len=1,
        output_new_cache_only=True,
        use_i64_token=False,
        output_cache=True
    ):
        super().__init__()
        self.model = Qwen2Model(config, ar_len, output_new_cache_only, use_i64_token)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_cache = output_cache

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
        self.output_conv.weight.data.copy_(self.lm_head.weight[:, :, None, None])

        del self.lm_head
        self.lm_head = forward_output_conv

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        hidden_states, output_k_cache, output_v_cache = self.model(
            tokens, atten_mask, input_pos, *args,
        )
        logits = self.lm_head(hidden_states)

        if self.output_cache:
            return logits, output_k_cache, output_v_cache
        return logits

    def get_example_inputs(self, use_kv_cache=True):
        model = self.model
        dtype = torch.int64 if model.use_i64_token else torch.int32
        tokens = torch.randint(
            self.vocab_size, (model.max_batch_size, model.ar_len), dtype=dtype
        )

        atten_mask = torch.full((model.ar_len, model.ar_len), torch.tensor(-255.0))
        mask_cond = torch.arange(atten_mask.size(-1))
        atten_mask.masked_fill_(
            mask_cond < (mask_cond + 1).view(atten_mask.size(-1), 1), 0
        )
        if model.max_seq_len != model.ar_len:
            atten_mask = torch.cat(
                [
                    torch.ones(model.ar_len, model.max_seq_len - model.ar_len) * -255.0,
                    atten_mask,
                ],
                dim=-1,
            )
        atten_mask = atten_mask[None, :, :].expand(
            model.max_batch_size, model.ar_len, model.max_seq_len
        )
        if use_kv_cache:
            pos_ids = torch.zeros((model.max_batch_size, model.ar_len), dtype=torch.int32)
            k_cache, v_cache = [], []

            for _ in range(model.n_layers):
                for _ in range(model.n_kv_heads):
                    # transpose first to decrease the runtime efforts
                    k_cache.append(
                        torch.zeros(
                            model.max_batch_size,
                            model.head_dim,
                            model.max_seq_len - model.ar_len,
                        )
                    )
                    v_cache.append(
                        torch.zeros(
                            model.max_batch_size,
                            model.max_seq_len - model.ar_len,
                            model.head_dim,
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
        model = self.model
        return {
            "get_ar_len": model.ar_len,
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_dim": model.dim,
            "get_head_dim": model.head_dim,
            "get_max_batch_size": model.max_batch_size,
            "get_max_seq_len": model.max_seq_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": model.n_kv_heads,
            "get_n_layers": model.n_layers,
            "get_vocab_size": model.vocab_size,
            "get_use_kv_cache": model.use_kv_cache,
        }
