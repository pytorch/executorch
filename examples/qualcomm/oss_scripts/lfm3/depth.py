# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static LFM3-Omni depth decoder: one speech frame's 12 codebook logits from the
backbone hidden state and the codes revealed so far (one MaskGIT pass)."""

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class _RMSNorm(nn.Module):
    """Explicit RMSNorm: QNN's fused rms_norm builder rejects some of these shapes."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, eps):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, n_heads, bias=False)
        self.q_norm = _RMSNorm(head_dim, eps)
        self.k_norm = _RMSNorm(head_dim, eps)

    def forward(self, x, cos, sin):
        n = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(n, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(n, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(n, self.n_kv_heads, self.head_dim)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.head_dim**-0.5, dim=-1)
        y = (attn @ v).transpose(0, 1)  # [n, heads, head_dim]
        y = y * torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        return self.o_proj(y.reshape(n, -1))


class _Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["norm_eps"]
        self.self_attn = _Attention(
            dim,
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["head_dim"],
            eps,
        )
        self.feed_forward = nn.ModuleDict(
            {
                "w1": nn.Linear(dim, cfg["intermediate_size"], bias=False),
                "w2": nn.Linear(cfg["intermediate_size"], dim, bias=False),
                "w3": nn.Linear(dim, cfg["intermediate_size"], bias=False),
            }
        )
        self.operator_norm = _RMSNorm(dim, eps)
        self.ffn_norm = _RMSNorm(dim, eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.operator_norm(x), cos, sin)
        f, y = self.feed_forward, self.ffn_norm(x)
        return x + f["w2"](F.silu(f["w1"](y)) * f["w3"](y))


class StaticDepthDecoder(nn.Module):
    """hidden [1, text_hidden], codes [n_cb] (int32), masked [n_cb] (1.0 = hidden)
    -> logits [n_cb, codebook_size]."""

    def __init__(
        self, depth_cfg: Dict, text_hidden: int, n_cb: int, codebook_size: int
    ):
        super().__init__()
        dim = depth_cfg["hidden_size"]
        self.n_cb, self.dim, self.codebook_size = n_cb, dim, codebook_size
        self.depth_linear = nn.Linear(text_hidden, n_cb * dim)
        self.mask_embedding = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList(
            [_Layer(depth_cfg) for _ in range(depth_cfg["num_hidden_layers"])]
        )
        self.embedding_norm = _RMSNorm(dim, depth_cfg["norm_eps"])
        # Per-codebook tables, stacked: input lookup, and output head with each
        # codebook's head-norm weight folded in (QNN needs 1-D rms_norm weights).
        self.tables = nn.Parameter(torch.zeros(n_cb, codebook_size, dim))
        self.out_tables = nn.Parameter(torch.zeros(n_cb, codebook_size, dim))
        self.head_eps = depth_cfg["norm_eps"]
        self.register_buffer(
            "offsets",
            torch.arange(n_cb, dtype=torch.int32) * codebook_size,
            persistent=False,
        )
        hd = depth_cfg["head_dim"]
        theta = depth_cfg["rope_parameters"]["full_attention"]["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
        freqs = torch.outer(torch.arange(n_cb).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)[:, None, :]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, hidden, codes, masked):
        x = self.depth_linear(hidden).view(self.n_cb, self.dim)
        code_emb = self.tables.view(-1, self.dim)[codes + self.offsets]
        m = masked.unsqueeze(-1)
        x = x + m * self.mask_embedding + (1.0 - m) * code_emb
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        x = self.embedding_norm(x).unsqueeze(1)  # [n_cb, 1, dim]
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.head_eps)
        return torch.bmm(x, self.out_tables.transpose(1, 2)).squeeze(1)


def depth_state_dict(
    hf_sd: Dict[str, torch.Tensor], n_cb: int
) -> Dict[str, torch.Tensor]:
    p = "model.depth_decoder."
    tables = torch.stack(
        [
            hf_sd[f"{p}depth_embeddings.{i}.embedding.weight"].float()
            for i in range(n_cb)
        ]
    )
    head_norm = torch.stack(
        [
            hf_sd[f"{p}depth_embeddings.{i}.embedding_norm.weight"].float()
            for i in range(n_cb)
        ]
    )[:, None, :]
    out = {
        "depth_linear.weight": hf_sd[p + "depth_linear.weight"].float(),
        "depth_linear.bias": hf_sd[p + "depth_linear.bias"].float(),
        "mask_embedding": hf_sd[p + "mask_embedding"].float(),
        "embedding_norm.weight": hf_sd[p + "model.embedding_norm.weight"].float(),
        "tables": tables,
        "out_tables": tables * head_norm,
    }
    for k, v in hf_sd.items():
        if k.startswith(p + "model.layers."):
            out[k[len(p + "model.") :]] = v.float()
    return out


class StaticFrameEmbedding(nn.Module):
    """codes [n_cb] (int32) -> decoder input embedding [1, 1, text_hidden]: the
    sum of the frame's codebook embeddings (one shared table, offset per codebook)."""

    def __init__(self, n_cb: int, codebook_size: int, text_hidden: int):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(n_cb * codebook_size, text_hidden))
        self.register_buffer(
            "offsets",
            torch.arange(n_cb, dtype=torch.int32) * codebook_size,
            persistent=False,
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.table[codes + self.offsets].sum(0).view(1, 1, -1)


def frame_embedding_state_dict(
    hf_sd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {"table": hf_sd["model.codebook_embedding.weight"].float()}
