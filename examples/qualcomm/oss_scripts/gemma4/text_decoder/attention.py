# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch

from executorch.examples.models.gemma4.text_decoder.gemma4_config import Gemma4Config
from executorch.examples.qualcomm.oss_scripts.llama.model import (
    NORM_REGISTRY,
    repeat_kv,
    ROTARY_EMB_REGISTRY,
)
from torch import nn


class Gemma4Attention(nn.Module):
    """Gemma 4 attention in static (flat KV I/O) style.

    Handles per-layer head_dim, QK-norm (before RoPE), scaleless V-norm,
    partial/full RoPE, MQA (n_kv_heads=1), and YOCO donor/shared roles.
    """

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
        output_new_cache_only: bool = True,
        enable_masked_softmax: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.get_head_dim(layer_idx)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.is_sliding = config.is_sliding_attention(layer_idx)
        self.is_kv_shared_layer = config.is_kv_shared_layer(layer_idx)
        self.is_kv_donor_layer = config.is_kv_donor_layer(layer_idx)
        self.output_new_cache_only = output_new_cache_only
        self.enable_masked_softmax = enable_masked_softmax

        # Gemma 4 uses scale=1.0; QK-norm handles normalisation
        self.scaling = 1.0

        # Q projection is present on every layer
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.q_norm = NORM_REGISTRY["rmsnorm"](self.head_dim, eps=config.rms_norm_eps)

        # KV projections and norms are always created (checkpoint compatibility).
        # For YOCO-shared layers these weights are loaded but not used in forward.
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.k_norm = NORM_REGISTRY["rmsnorm"](self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = nn.RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.apply_rope = (
            ROTARY_EMB_REGISTRY["default"]
            if self.is_sliding
            else ROTARY_EMB_REGISTRY["partial"]
        )

        self.attn_softmax = nn.Softmax(dim=-1)

    def prepare_attention_conv(self):
        self.q_proj_conv = nn.Conv2d(
            self.hidden_size, self.num_heads * self.head_dim, 1, bias=False
        )
        self.k_proj_conv = nn.Conv2d(
            self.hidden_size, self.num_kv_heads * self.head_dim, 1, bias=False
        )
        self.v_proj_conv = nn.Conv2d(
            self.hidden_size, self.num_kv_heads * self.head_dim, 1, bias=False
        )
        self.o_proj_conv = nn.Conv2d(
            self.num_heads * self.head_dim, self.hidden_size, 1, bias=False
        )

        self.forward_no_conv = self.forward
        self.forward = self.forward_attention_conv

        self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
        self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
        self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
        self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

        del self.q_proj
        del self.k_proj
        del self.v_proj
        del self.o_proj

    def forward_attention_conv(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        donor_k: Optional[torch.Tensor] = None,
        donor_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape
        hs = torch.reshape(
            hidden_states, (bsz, seq_len, 1, self.hidden_size)
        ).transpose(1, 3)

        q = self.q_proj_conv(hs)
        q = q.permute(0, 3, 1, 2).squeeze(-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        if self.is_kv_shared_layer:
            q = self.apply_rope(q, freqs_cos, freqs_sin)
            kh = donor_k
            vh = donor_v
            new_k, new_v = None, None
        else:
            k = self.k_proj_conv(hs)
            v = self.v_proj_conv(hs)
            k = k.permute(0, 3, 1, 2).squeeze(-1)
            v = v.permute(0, 3, 1, 2).squeeze(-1)
            k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k = self.k_norm(k)
            v = self.v_norm(v)

            q = self.apply_rope(q, freqs_cos, freqs_sin)
            k = self.apply_rope(k, freqs_cos, freqs_sin)

            k = k.transpose(2, 3)

            if k_cache is not None:
                kh = torch.cat([k_cache, k], dim=-1)
                vh = torch.cat([v_cache, v], dim=2)
            else:
                kh = k
                vh = v

            new_k = k if self.output_new_cache_only else kh
            new_v = v if self.output_new_cache_only else vh

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        attn = attn * self.scaling
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)

        y = attn @ vh
        y = y.transpose(1, 2)
        y = y.reshape(bsz, seq_len, 1, -1).transpose(1, 3)
        y = self.o_proj_conv(y)
        y = y.transpose(1, 3)
        y = y.reshape(bsz, seq_len, -1)

        return y, new_k, new_v

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        donor_k: Optional[torch.Tensor] = None,
        donor_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        K cache layout (static_llama convention): [b, n_kv, head_dim, ctx]
        V cache layout:                           [b, n_kv, ctx, head_dim]

        Returns (output, new_k, new_v):
          - self-decoder / donor layers: new_k/new_v hold the new cache slice
            (or full cache if output_new_cache_only=False).
          - non-donor shared layers: (None, None) — no KV I/O.
        """
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        if self.is_kv_shared_layer:
            q = self.apply_rope(q, freqs_cos, freqs_sin)
            kh = donor_k
            vh = donor_v
            new_k, new_v = None, None
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k = self.k_norm(k)
            v = self.v_norm(v)

            q = self.apply_rope(q, freqs_cos, freqs_sin)
            k = self.apply_rope(k, freqs_cos, freqs_sin)

            k = k.transpose(2, 3)

            if k_cache is not None:
                kh = torch.cat([k_cache, k], dim=-1)
                vh = torch.cat([v_cache, v], dim=2)
            else:
                kh = k
                vh = v

            new_k = k if self.output_new_cache_only else kh
            new_v = v if self.output_new_cache_only else vh

        kh = repeat_kv(kh, self.num_key_value_groups)
        vh = repeat_kv(vh, self.num_key_value_groups)

        attn = q @ kh
        attn = attn * self.scaling
        if self.enable_masked_softmax:
            attn_min = torch.amin(attn, dim=-1, keepdim=True)
            minus_value = -20
            attn = torch.where(atten_mask == 0, attn, attn_min + minus_value)
        else:
            attn = attn + atten_mask
        attn = self.attn_softmax(attn)

        y = attn @ vh
        y = y.transpose(1, 2).reshape(bsz, seq_len, -1)
        y = self.o_proj(y)

        return y, new_k, new_v
