# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exportable Muse Glimmer vision encoder.

Data-dependent preprocessing runs on the host, leaving only ``num_patches``
dynamic in the device graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.examples.models.muse_glimmer.model.model import (
    RMSNormNoWeight,
    rotate_interleaved,
)
from executorch.examples.models.muse_glimmer.vision.precompute import (
    MuseGlimmerVisionConfig,
)


# Attention / MLP / block


class MuseGlimmerVisionAttention(nn.Module):
    """Bidirectional multi-head attention with 2D RoPE (bias-ful projections)."""

    def __init__(self, config: MuseGlimmerVisionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        dim = config.latent_dim
        qkv_dim = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(dim, qkv_dim, bias=True)
        self.k_proj = nn.Linear(dim, qkv_dim, bias=True)
        self.v_proj = nn.Linear(dim, qkv_dim, bias=True)
        self.o_proj = nn.Linear(qkv_dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,  # [1, P, dim]
        cos: torch.Tensor,  # [1, P, 1, head_dim//2]
        sin: torch.Tensor,  # [1, P, 1, head_dim//2]
        attn_mask: torch.Tensor,  # [1, 1, P, P] bool
    ) -> torch.Tensor:
        B, P, _ = x.shape
        q = self.q_proj(x).view(B, P, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, P, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, P, self.n_heads, self.head_dim)

        # 2D RoPE (real interleaved form). Small grid positions + fp32-precomputed
        # cos/sin make fp16 rotation safe, so opt into native-dtype rotation.
        q = rotate_interleaved(q, cos, sin, allow_low_precision=True)
        k = rotate_interleaved(k, cos, sin, allow_low_precision=True)

        q = q.transpose(1, 2)  # [B, H, P, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Default SDPA scale (1/sqrt(head_dim)) matches the eager encoder.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, P, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MuseGlimmerVisionMLP(nn.Module):
    """Bias-enabled feed-forward block used by the vision encoder."""

    def __init__(self, config: MuseGlimmerVisionConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.latent_dim, config.mlp_hidden, bias=True)
        self.c_proj = nn.Linear(config.mlp_hidden, config.latent_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class MuseGlimmerVisionBlock(nn.Module):
    """Pre-LayerNorm vision transformer block."""

    def __init__(self, config: MuseGlimmerVisionConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.latent_dim)
        self.attn = MuseGlimmerVisionAttention(config)
        self.ln_2 = nn.LayerNorm(config.latent_dim)
        self.mlp = MuseGlimmerVisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# Top-level encoder (encoder + adapter + projection + perception norm)


class MuseGlimmerVisionEncoder(nn.Module):
    """End-to-end image branch: host-precomputed patch inputs -> text-space embeds.

    Submodule names match the mmproj GGUF mapping. The positional-embedding
    table is loaded separately for host preprocessing.
    """

    def __init__(self, config: MuseGlimmerVisionConfig | None = None):
        super().__init__()
        if config is None:
            config = MuseGlimmerVisionConfig()
        self.config = config
        self.downsample_factor = config.downsample_factor

        self.conv1_linear = nn.Linear(config.patch_dim, config.latent_dim, bias=False)
        self.ln_pre = nn.LayerNorm(config.latent_dim)
        self.blocks = nn.ModuleList(
            [MuseGlimmerVisionBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_post = nn.LayerNorm(config.latent_dim)

        self.adapter_fc = nn.Linear(
            config.encoder_output_dim, config.adapter_dim, bias=False
        )
        self.adapter_proj = nn.Linear(
            config.adapter_dim, config.adapter_dim, bias=False
        )
        self.vision_proj = nn.Linear(config.adapter_dim, config.hidden_size, bias=False)
        self.perception_emb_norm = RMSNormNoWeight(
            config.hidden_size, config.rms_norm_eps
        )

        # Static per-layer routing (global vs sparse attention), resolved at
        # trace time.
        self._is_global = [config.layer_is_global(i) for i in range(config.n_layers)]

    def forward(
        self,
        patches: torch.Tensor,  # [1, P, patch_dim] f32
        pos_emb: torch.Tensor,  # [1, P, latent] bf16
        cos_2d: torch.Tensor,  # [P, head_dim//2] f32
        sin_2d: torch.Tensor,  # [P, head_dim//2] f32
        sparse_perm: torch.Tensor,  # [P] i64
        inv_perm: torch.Tensor,  # [P] i64
        global_mask: torch.Tensor,  # [1, 1, P, P] bool
        sparse_mask: torch.Tensor,  # [1, 1, P, P] bool
        pixel_perm: torch.Tensor,  # [P] i64
    ) -> torch.Tensor:  # [1, P//f^2, hidden_size] activation dtype
        w_dtype = self.conv1_linear.weight.dtype

        # Patch embed + host-interpolated positional embedding + pre-norm.
        x = self.conv1_linear(patches.to(w_dtype))
        x = x + pos_emb.to(w_dtype)
        x = self.ln_pre(x)

        # Sparse tiling: reorder tokens (and RoPE) so each 32x32 block is
        # contiguous; global layers see all tokens, sparse layers see one block.
        x = x.index_select(1, sparse_perm)
        cos_p = cos_2d.index_select(0, sparse_perm)
        sin_p = sin_2d.index_select(0, sparse_perm)
        # [P, head_dim//2] -> [1, P, 1, head_dim//2] for rotate_interleaved.
        cos_b = cos_p.unsqueeze(0).unsqueeze(2)
        sin_b = sin_p.unsqueeze(0).unsqueeze(2)

        for i, block in enumerate(self.blocks):
            mask = global_mask if self._is_global[i] else sparse_mask
            x = block(x, cos_b, sin_b, mask)

        # Undo the sparse permutation, then final norm.
        x = x.index_select(1, inv_perm)
        x = self.ln_post(x)

        # Pixel-shuffle 2x downsample: gather then (n_out, f*f, d)->(n_out, d*f*f).
        f = self.downsample_factor
        d = self.config.latent_dim
        x_flat = x[0].index_select(0, pixel_perm)  # [P, d]
        x_ds = (
            x_flat.view(-1, f * f, d)
            .permute(0, 2, 1)
            .contiguous()
            .view(1, -1, d * f * f)
        )  # [1, P//f^2, d*f^2]

        # Adapter -> projection into text space -> perception norm.
        x_ds = F.gelu(self.adapter_fc(x_ds))
        x_ds = F.gelu(self.adapter_proj(x_ds))
        x_ds = self.vision_proj(x_ds)
        x_ds = self.perception_emb_norm(x_ds)
        return x_ds.to(w_dtype)
