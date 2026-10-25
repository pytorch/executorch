# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license in the root LICENSE.

"""Packed Gemma 4 dense decoder with runtime-owned sequence KV."""

import json

import executorch.extension.llm.cache.update_and_attend  # noqa: F401
import torch
from executorch.examples.models.gemma4.text_decoder import apply_rotary_emb
from executorch.examples.models.gemma4_31b.model import Gemma4_31B, Gemma4Attention
from executorch.extension.llm.export.model_metadata import write_cache_geometry
from torch import nn


class OffgraphGemmaAttention(nn.Module):
    def __init__(self, attention: Gemma4Attention, window: int):
        super().__init__()
        self.layer_idx = attention.layer_idx
        self.n_heads = attention.n_heads
        self.n_kv_heads = attention.n_kv_heads
        self.head_dim = attention.head_dim
        self.k_eq_v = attention.k_eq_v
        self.window = window
        self.scaling = attention.scaling
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        if not self.k_eq_v:
            self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.v_norm = attention.v_norm
        self.register_buffer("inv_freq", attention.inv_freq, persistent=False)

    def forward(self, x, input_pos, attn_mask=None):
        batch, width, _ = x.shape
        q = self.q_proj(x).view(batch, width, self.n_heads, self.head_dim)
        raw_k = self.k_proj(x).view(batch, width, self.n_kv_heads, self.head_dim)
        raw_v = raw_k if self.k_eq_v else self.v_proj(x).view(
            batch, width, self.n_kv_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(raw_k).transpose(1, 2)
        v = self.v_norm(raw_v).transpose(1, 2)
        freqs = torch.outer(input_pos.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        q, k = apply_rotary_emb(q, k, torch.cos(emb), torch.sin(emb))
        y = torch.ops.kvcache.update_and_attend(
            q, k, v, input_pos.reshape(-1, 1), self.layer_idx, self.scaling, q.dtype
        )
        y = y.transpose(1, 2).contiguous().view(batch, width, self.n_heads * self.head_dim)
        return self.o_proj(y)


class PackedGemma4(nn.Module):
    def __init__(self, model: Gemma4_31B):
        super().__init__()
        self.config = model.config
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.register_buffer("embed_normalizer", model.embed_normalizer, persistent=False)
        self.register_buffer("logit_softcap", model.logit_softcap, persistent=False)
        for layer in self.layers:
            layer.self_attn = OffgraphGemmaAttention(
                layer.self_attn, self.config.sliding_window if layer.is_sliding else 0
            )

    def forward(self, tokens, input_pos, logits_to_keep):
        x = self.embed_tokens(tokens) * self.embed_normalizer
        for layer in self.layers:
            x = layer(x, input_pos, None, None)
        x = self.norm(x[:, logits_to_keep, :])
        logits = self.lm_head(x).float()[0]
        cap = self.logit_softcap.float()
        return torch.tanh(logits / cap) * cap

    def sequence_manifest(self, max_write):
        return json.dumps({
            "version": 1,
            "maximum_capacity": self.config.max_seq_len,
            "max_write": max_write,
            "layers": [{"layer_id": layer.self_attn.layer_idx,
                        "policy": "ring" if layer.self_attn.window else "flat",
                        "window": layer.self_attn.window} for layer in self.layers],
        }, sort_keys=True, separators=(",", ":"))

    def cache_geometry(self):
        return write_cache_geometry(
            [layer.self_attn.n_kv_heads for layer in self.layers],
            [layer.self_attn.head_dim for layer in self.layers],
            [layer.self_attn.window for layer in self.layers],
        )
