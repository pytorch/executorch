# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
import torch.nn as nn

from executorch.examples.models.llama2.llama_transformer import (
    FeedForward,
    ModelArgs,
    precompute_freqs_cis,
    RMSNorm,
)


def apply_rotary_emb_single(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    x_r, x_i = x[..., ::2], x[..., 1::2]

    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_out = torch.cat([x_out_r, x_out_i], dim=-1)
    return x_out


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = config.n_heads // self.n_kv_heads
        self.max_seq_len = config.max_seq_len
        self.output_new_cache_only = output_new_cache_only

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        self.scale = float(self.head_dim) ** 0.5

    def prepare_sha(self):
        self.wq_sha = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
                for _ in range(self.n_heads)
            ]
        )
        self.wk_sha = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
                for _ in range(self.n_heads)
            ]
        )
        self.wv_sha = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
                for _ in range(self.n_heads)
            ]
        )

        self.forward_mha = self.forward
        self.forward = self.forward_sha

        for i in range(self.n_heads):
            self.wq_sha[i].weight.data.copy_(
                self.wq.weight[i * self.head_dim : (i + 1) * self.head_dim]
            )
            self.wk_sha[i].weight.data.copy_(
                self.wk.weight[i * self.head_dim : (i + 1) * self.head_dim]
            )
            self.wv_sha[i].weight.data.copy_(
                self.wv.weight[i * self.head_dim : (i + 1) * self.head_dim]
            )

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = [wq_sha(hidden_states) for wq_sha in self.wq_sha]
        k = [wk_sha(hidden_states) for wk_sha in self.wk_sha]
        v = [wv_sha(hidden_states) for wv_sha in self.wv_sha]
        for i in range(len(q)):
            q[i] = apply_rotary_emb_single(q[i], freqs_cos, freqs_sin)
            k[i] = apply_rotary_emb_single(k[i], freqs_cos, freqs_sin).permute(0, 2, 1)

        output_kh, output_vh, output_y = [], [], []
        for i, _ in enumerate(k_caches):
            # cat at the seq dim
            kh = torch.cat([k_caches[i], k[i]], dim=-1)
            vh = torch.cat([v_caches[i], v[i]], dim=1)

            attn = q[i] @ kh
            attn = attn / self.scale + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ vh

            if self.output_new_cache_only:
                output_kh.append(k[i])
                output_vh.append(v[i])
            else:
                output_kh.append(kh)
                output_vh.append(vh)
            output_y.append(y)

        y = torch.concat(output_y, dim=-1)
        y = self.wo(y)
        return y, output_kh, output_vh

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = hidden_states.shape

        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb_single(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb_single(k, freqs_cos, freqs_sin).permute(0, 2, 3, 1)

        output_kh, output_vh, output_y = [], [], []

        for i, _ in enumerate(k_caches):
            # cat at the seq dim
            kh = torch.cat([k_caches[i], k[:, i, :, :]], dim=-1)
            vh = torch.cat([v_caches[i], v[:, :, i, :]], dim=1)

            attn = q[:, :, i, :] @ kh
            attn = attn / self.scale + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ vh

            if self.output_new_cache_only:
                output_kh.append(k[:, i, :, :])
                output_vh.append(v[:, :, i, :])
            else:
                output_kh.append(kh)
                output_vh.append(vh)
            output_y.append(y)

        y = torch.concat(output_y, dim=-1)
        y = self.wo(y)

        return y, output_kh, output_vh


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, output_new_cache_only=False):
        super().__init__()
        self.dim = config.dim
        self.attention = LlamaAttention(
            config=config, output_new_cache_only=output_new_cache_only
        )
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

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
        h = x + h
        output = h + self.feed_forward(self.ffn_norm(h))
        return output, k_cache, v_cache


class LlamaModel(nn.Module):
    def __init__(self, config: ModelArgs, output_new_cache_only=True):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.rope_freq_base = config.rope_freq_base
        self.output_new_cache_only = output_new_cache_only

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, self.output_new_cache_only)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_freq_base,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        atten_mask: torch.Tensor,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = self.freqs_cos[input_pos][0]
        freqs_sin = self.freqs_sin[input_pos][0]

        hidden_states = self.tok_embeddings(tokens)
        for ind, decoder_layer in enumerate(self.layers):
            offset_k = ind * self.n_heads
            offset_v = self.n_layers * self.n_heads + offset_k
            k_caches = args[offset_k : offset_k + self.n_heads]
            v_caches = args[offset_v : offset_v + self.n_heads]
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

        return logits, output_k_cache, output_v_cache

    def get_example_inputs(self):
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, 1), dtype=torch.int32
        )
        pos_ids = torch.zeros((self.max_batch_size, 1), dtype=torch.int32)
        k_cache, v_cache = [], []
        atten_mask = torch.full((self.max_batch_size, self.max_seq_len), -255.0)
        atten_mask[:, -1] = 0
        for _ in range(self.n_layers):
            for _ in range(self.n_heads):
                # transpose first to decrease the runtime efforts
                k_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.head_dim,
                        self.max_seq_len - 1,
                    )
                )
                v_cache.append(
                    torch.zeros(
                        self.max_batch_size,
                        self.max_seq_len - 1,
                        self.head_dim,
                    )
                )
        return (
            tokens,
            pos_ids,
            atten_mask,
            k_cache,
            v_cache,
        )

    def get_export_inputs(self):
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, 1), dtype=torch.int32
        )
        pos_ids = torch.zeros((self.max_batch_size, 1), dtype=torch.int32)
        # this is important for torch.export not to take it as dummy input
        k_cache, v_cache = [], []
        atten_mask = torch.full((self.max_batch_size, self.max_seq_len), -255.0)
        atten_mask[:, -1] = 0
        for _ in range(self.n_layers):
            for _ in range(self.n_heads):
                # transpose first to decrease the runtime efforts
                k_cache.append(
                    torch.randn(
                        self.max_batch_size,
                        self.head_dim,
                        self.max_seq_len - 1,
                    )
                )
                v_cache.append(
                    torch.randn(
                        self.max_batch_size,
                        self.max_seq_len - 1,
                        self.head_dim,
                    )
                )
        return (
            tokens,
            pos_ids,
            atten_mask,
            k_cache,
            v_cache,
        )

    def get_metadata(self):
        # TODO: modify this when enabling LLAMA 7B
        return {
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_dim": self.dim,
            "get_head_dim": self.dim // self.n_heads,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
        }
