# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
import torch.nn as nn

from executorch.examples.models.llama2.llama_transformer import (
    apply_rotary_emb,
    FeedForward,
    ModelArgs,
    precompute_freqs_cis,
    RMSNorm,
)


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelArgs, split_kv_cache=False):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = config.n_heads // self.n_kv_heads
        self.max_seq_len = config.max_seq_len
        self.split_kv_cache = split_kv_cache

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        scale = float(self.head_dim) ** -0.5
        scale_tensor = torch.tensor(
            [scale], dtype=torch.float32, requires_grad=False
        ).view(1, 1, 1)
        self.register_buffer("scale_tensor", scale_tensor, False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        kv_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = hidden_states.shape

        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        if self.split_kv_cache:
            output_kh, output_vh, output_y = [], [], []
            for i, _ in enumerate(k_caches):
                kh = k_caches[i] + k[:, :, i, :] * kv_mask
                vh = v_caches[i] + v[:, :, i, :] * kv_mask

                attn = q[:, :, i, :] @ kh.permute(0, 2, 1)
                attn = attn * self.scale_tensor + atten_mask
                attn = self.attn_softmax(attn)
                y = attn @ vh

                output_kh.append(kh)
                output_vh.append(vh)
                output_y.append(y)

            y = torch.concat(output_y, dim=-1)
            y = self.wo(y)
            return y, output_kh, output_vh
        else:
            k = k_caches + k * kv_mask
            v = v_caches + v * kv_mask

            attn = q.transpose(1, 2) @ k.permute(0, 2, 3, 1)
            attn = attn * self.scale_tensor + atten_mask
            attn = self.attn_softmax(attn)
            y = attn @ v.transpose(1, 2)
            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
            y = self.wo(y)

            return y, k, v


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, split_kv_cache=False):
        super().__init__()
        self.dim = config.dim
        self.attention = LlamaAttention(config=config, split_kv_cache=split_kv_cache)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        kv_mask: torch.Tensor,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        h, k_cache, v_cache = self.attention(
            hidden_states=self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            atten_mask=atten_mask,
            kv_mask=kv_mask,
            k_caches=k_caches,
            v_caches=v_caches,
        )
        h = x + h
        output = h + self.feed_forward(self.ffn_norm(h))
        return output, k_cache, v_cache


class LlamaModel(nn.Module):
    def __init__(self, config: ModelArgs, split_kv_cache=False):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_layers = config.n_layers
        self.vocab_size = config.vocab_size
        self.split_kv_cache = split_kv_cache

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, split_kv_cache) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_freq_base,
        )
        atten_mask = torch.triu(
            torch.full(
                (self.max_seq_len, self.max_seq_len),
                -255.0,
            ),
            diagonal=1,
        )
        self.register_buffer("atten_mask", atten_mask, persistent=False)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        if split_kv_cache:
            self.register_buffer("mask", torch.ones(self.head_dim), persistent=False)
            self.register_buffer("unmask", torch.zeros(self.head_dim), persistent=False)
        else:
            self.register_buffer("mask", torch.ones(self.dim), persistent=False)
            self.register_buffer("unmask", torch.zeros(self.dim), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        kv_mask: torch.Tensor,
        *args,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        # following tensors should be invariant across batches
        freqs_cos = self.freqs_cos[input_pos][0]
        freqs_sin = self.freqs_sin[input_pos][0]
        atten_mask = self.atten_mask[input_pos][0]

        hidden_states = self.tok_embeddings(tokens)
        for ind, decoder_layer in enumerate(self.layers):
            if self.split_kv_cache:
                offset_k = ind * self.n_heads
                offset_v = self.n_layers * self.n_heads + offset_k
                k_caches = args[offset_k : offset_k + self.n_heads]
                v_caches = args[offset_v : offset_v + self.n_heads]
                hidden_states, k, v = decoder_layer(
                    hidden_states,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    atten_mask=atten_mask,
                    kv_mask=kv_mask,
                    k_caches=k_caches,
                    v_caches=v_caches,
                )
                output_k_cache.extend(k)
                output_v_cache.extend(v)
            else:
                k_caches = args[ind]
                v_caches = args[self.n_layers + ind]
                hidden_states, k, v = decoder_layer(
                    hidden_states,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    atten_mask=atten_mask,
                    kv_mask=kv_mask.view(
                        self.max_seq_len, self.n_kv_heads, self.head_dim
                    ),
                    k_caches=k_caches,
                    v_caches=v_caches,
                )
                output_k_cache.append(k)
                output_v_cache.append(v)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        # TODO: add op builder for kv mask update once HTP supports more ops
        #       this part is now expected to be fallback on cpu
        # for simplicity, input_pos is assumed to never go over max_seq_len-1
        kv_mask[input_pos] = self.unmask
        kv_mask[input_pos + 1] = self.mask

        return logits, kv_mask, output_k_cache, output_v_cache

    def get_example_inputs(self):
        tokens = torch.randint(
            self.vocab_size, (self.max_batch_size, 1), dtype=torch.int32
        )
        pos_ids = torch.zeros((self.max_batch_size, 1), dtype=torch.int32)
        k_cache, v_cache = [], []
        if self.split_kv_cache:
            kv_mask = torch.zeros(self.max_seq_len, self.head_dim)
            kv_mask[0] = torch.ones(self.head_dim)
            for _ in range(self.n_layers):
                for _ in range(self.n_heads):
                    k_cache += torch.zeros(
                        self.max_batch_size,
                        self.max_seq_len,
                        self.head_dim,
                    )
                    v_cache += torch.zeros(
                        self.max_batch_size,
                        self.max_seq_len,
                        self.head_dim,
                    )
        else:
            kv_mask = torch.zeros(self.max_seq_len, self.dim)
            kv_mask[0] = torch.ones(self.dim)
            for _ in range(self.n_layers):
                k_cache += torch.zeros(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_heads,
                    self.head_dim,
                )
                v_cache += torch.zeros(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_heads,
                    self.head_dim,
                )
        return (
            tokens,
            pos_ids,
            kv_mask,
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
        if self.split_kv_cache:
            kv_mask = torch.zeros(self.max_seq_len, self.head_dim)
            kv_mask[0] = torch.ones(self.head_dim)
            for _ in range(self.n_layers):
                for _ in range(self.n_heads):
                    k_cache += torch.randn(
                        self.max_batch_size,
                        self.max_seq_len,
                        self.head_dim,
                    )
                    v_cache += torch.randn(
                        self.max_batch_size,
                        self.max_seq_len,
                        self.head_dim,
                    )
        else:
            kv_mask = torch.zeros(self.max_seq_len, self.dim)
            kv_mask[0] = torch.ones(self.dim)
            for _ in range(self.n_layers):
                k_cache += torch.randn(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_heads,
                    self.head_dim,
                )
                v_cache += torch.randn(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_heads,
                    self.head_dim,
                )
        return (
            tokens,
            pos_ids,
            kv_mask,
            k_cache,
            v_cache,
        )

    def get_metadata(self):
        # TODO: modify this when enabling LLAMA 7B
        return {
            "get_bos_id": 1,
            "get_eos_id": 2,
            "get_head_dim": self.dim // self.n_heads,
            "get_max_batch_size": self.max_batch_size,
            "get_max_seq_len": self.max_seq_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": self.n_heads,
            "get_n_layers": self.n_layers,
            "get_vocab_size": self.vocab_size,
        }
