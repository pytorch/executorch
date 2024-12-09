# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Example script for exporting Llama2 to flatbuffer

import math
from typing import List, Optional, Tuple

import torch
from executorch.examples.models.llama.llama_transformer import Attention
from torch import nn


def apply_rotary_emb_single(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    x_r, x_i = x[..., ::2], x[..., 1::2]

    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_out = torch.cat([x_out_r, x_out_i], dim=-1)
    return x_out


class KVCacheSHA(torch.nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.float32,
    ):
        super().__init__()

        # a buffer per head
        cache_shape = (max_batch_size, max_seq_length, head_dim)
        for i in range(n_heads):
            self.register_buffer(
                f"past_k_caches_{i}",
                torch.zeros(cache_shape, dtype=dtype, device="cpu"),
                persistent=False,
            )
            self.register_buffer(
                f"past_v_caches_{i}",
                torch.zeros(cache_shape, dtype=dtype, device="cpu"),
                persistent=False,
            )

    def update(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        cache_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_k = torch.ops.aten.index_put_(
            getattr(self, f"past_k_caches_{cache_idx}"), [None, input_pos], k_val
        )
        new_v = torch.ops.aten.index_put_(
            getattr(self, f"past_v_caches_{cache_idx}"), [None, input_pos], v_val
        )
        return new_k, new_v

    def get_cache(self, head_idx):
        return getattr(self, f"past_k_caches_{head_idx}"), getattr(
            self, f"past_v_caches_{head_idx}"
        )


class SDPASHA(torch.nn.Module):

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        n_rep: int,
        head_dim: int,
        dim: int,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_rep = n_rep
        self.dim = dim
        self.kv_cache = KVCacheSHA(
            max_batch_size, max_seq_length, n_heads // n_rep, head_dim
        )
        self.scale_factor = math.sqrt(head_dim)

    def forward(
        self,
        input_pos: torch.Tensor,
        qs: List[torch.Tensor],
        ks: List[torch.Tensor],
        vs: List[torch.Tensor],
        mask,
    ):

        transpose_ks = []
        for i in range(len(ks)):
            new_k, _ = self.kv_cache.update(input_pos, ks[i], vs[i], i)
            transpose_ks.append(new_k.transpose(-2, -1).contiguous())

        output = []
        for i, q in enumerate(qs):
            cache_idx = i // self.n_rep
            _, v = self.kv_cache.get_cache(cache_idx)

            attn_mask = mask[input_pos]

            attn_weight = q @ transpose_ks[cache_idx] / self.scale_factor
            attn_weight += attn_mask
            attn_weight = torch.softmax(attn_weight, dim=-1)
            output.append(attn_weight @ v.contiguous())

        return torch.cat(output, dim=-1)


class AttentionSHA(nn.Module):
    def __init__(self, attention_mha: nn.Module):
        super().__init__()
        if not attention_mha.use_kv_cache:
            raise NotImplementedError("bert mode is not support")

        self.n_heads = attention_mha.n_heads
        self.n_kv_heads = attention_mha.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dim = attention_mha.dim
        self.max_batch_size = attention_mha.max_batch_size
        self.max_seq_len = attention_mha.max_seq_len
        self.head_dim = attention_mha.dim // self.n_heads
        self.SDPA = SDPASHA(
            self.max_batch_size,
            self.max_seq_len,
            self.n_heads,
            self.n_rep,
            self.head_dim,
            self.dim,
        )
        self.wq = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
                for _ in range(self.n_heads)
            ]
        )
        self.wk = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
                for _ in range(self.n_kv_heads)
            ]
        )
        self.wv = nn.ModuleList(
            [
                nn.Linear(self.dim, self.head_dim, bias=False)
                for _ in range(self.n_kv_heads)
            ]
        )

        for i in range(self.n_heads):
            self.wq[i].weight.data.copy_(
                # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
                #  attribute `weight`.
                attention_mha.wq.weight[i * self.head_dim : (i + 1) * self.head_dim]
            )
        for i in range(self.n_kv_heads):
            self.wk[i].weight.data.copy_(
                # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
                #  attribute `weight`.
                attention_mha.wk.weight[i * self.head_dim : (i + 1) * self.head_dim]
            )
            self.wv[i].weight.data.copy_(
                # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no
                #  attribute `weight`.
                attention_mha.wv.weight[i * self.head_dim : (i + 1) * self.head_dim]
            )
        self.wo = attention_mha.wo

        causal_mask = torch.tril(
            torch.ones(
                self.max_seq_len,
                self.max_seq_len,
                dtype=torch.bool,
                device="cpu",
            )
        )
        self.register_buffer("mask", causal_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ):
        # QKV
        q = [wq(x) for wq in self.wq]
        k = [wk(x) for wk in self.wk]
        v = [wv(x) for wv in self.wv]
        for i in range(len(q)):
            q[i] = apply_rotary_emb_single(q[i], freqs_cos, freqs_sin)
        for i in range(len(k)):
            k[i] = apply_rotary_emb_single(k[i], freqs_cos, freqs_sin)

        output = self.SDPA(input_pos, q, k, v, self.mask)
        return self.wo(output)


def replace_attention_to_attention_sha(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, Attention):
            setattr(
                module,
                name,
                AttentionSHA(child),
            )
        else:
            replace_attention_to_attention_sha(child)
    return module
