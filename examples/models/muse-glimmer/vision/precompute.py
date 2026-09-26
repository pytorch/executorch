# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side preprocessing for the Muse Glimmer vision encoder.

This module computes the data-dependent inputs to the exported graph and is
mirrored by ``vision/preprocess.h`` for the C++ runtime.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# Configuration from the model and mmproj metadata.


@dataclass
class MuseGlimmerVisionConfig:
    patch_size: int = 14
    patch_temporal: int = 2
    downsample_factor: int = 2
    sparse_attention_factor: int = 4
    pos_emb_grid_h: int = 32
    pos_emb_grid_w: int = 32
    latent_dim: int = 1536
    n_heads: int = 16
    n_layers: int = 50
    rope_theta: float = 10000.0
    mlp_hidden: int = 8960
    adapter_dim: int = 4096
    encoder_output_dim: int = 6144  # latent_dim * downsample_factor**2
    hidden_size: int = 6656  # text hidden size (vision_projection output)
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.latent_dim // self.n_heads  # 96

    @property
    def patch_dim(self) -> int:
        return self.patch_temporal * 3 * self.patch_size**2  # 1176

    def layer_is_global(self, layer_idx: int) -> bool:
        sf = self.sparse_attention_factor
        return (layer_idx == self.n_layers - 1) or ((layer_idx + 1) % sf == 0)


# Individual preprocessing steps.


def patchify_image(
    image: torch.Tensor,  # [3, H, W] normalized to [-1, 1], any float dtype
    config: MuseGlimmerVisionConfig,
) -> tuple[torch.Tensor, int, int]:
    """Unfold a normalized image into patch vectors (mirrors the c==3 path).

    Returns ``(patches [1, P, patch_dim] float32, grid_h, grid_w)`` where
    ``patch_dim = patch_temporal * 3 * patch_size**2`` and the temporal axis is
    a replicate of the single frame (image path).
    """
    ps = config.patch_size
    pt = config.patch_temporal
    if image.ndim == 3:
        image = image.unsqueeze(0)  # [1, 3, H, W]
    _, c, h, w = image.shape
    if c != 3:
        raise ValueError(f"patchify_image expects 3 channels, got {c}")
    if h % ps != 0 or w % ps != 0:
        raise ValueError(f"image {h}x{w} not divisible by patch_size {ps}")
    grid_h, grid_w = h // ps, w // ps
    n_tokens = grid_h * grid_w

    patches = image.unfold(2, ps, ps).unfold(3, ps, ps)
    patches = patches.contiguous().view(1, c, grid_h, grid_w, ps, ps)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [1,gh,gw,c,ps,ps]
    patches = patches.unsqueeze(3).expand(-1, -1, -1, pt, -1, -1, -1)
    patches = patches.reshape(1, n_tokens, -1)  # [1, n_tokens, pt*c*ps*ps]
    return patches.to(torch.float32), grid_h, grid_w


def interpolate_pos_emb(
    pos_embed_table: torch.Tensor,  # [pos_grid_h*pos_grid_w, latent_dim] bf16
    grid_h: int,
    grid_w: int,
    config: MuseGlimmerVisionConfig,
) -> torch.Tensor:
    """Interpolate the positional table to ``grid_h`` by ``grid_w``.

    Interpolation stays in the table's native dtype to preserve reference
    numerics. Returns ``[1, grid_h*grid_w, latent_dim]``.
    """
    gh, gw = config.pos_emb_grid_h, config.pos_emb_grid_w
    latent = config.latent_dim
    dtype = pos_embed_table.dtype
    pos_emb = (
        pos_embed_table.view(gh, gw, latent).permute(2, 0, 1).unsqueeze(0)
    )  # [1, latent, gh, gw]
    inv_h = 1.0 / grid_h
    inv_w = 1.0 / grid_w
    ys = torch.linspace(-1 + inv_h, 1 - inv_h, grid_h, dtype=dtype)
    xs = torch.linspace(-1 + inv_w, 1 - inv_w, grid_w, dtype=dtype)
    pos_xy = torch.stack(torch.meshgrid(ys, xs, indexing="xy"), dim=-1).reshape(-1, 2)[
        None, None
    ]
    sampled = F.grid_sample(pos_emb, pos_xy, mode="bilinear", align_corners=False)
    pos = sampled[0, :, 0, :].T  # [n_tokens, latent]
    return pos.unsqueeze(0).contiguous()  # [1, n_tokens, latent]


def make_2d_rope(
    grid_h: int,
    grid_w: int,
    config: MuseGlimmerVisionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real 2D RoPE cos/sin tensors for interleaved rotation.

    The result is numerically equivalent to the reference complex formulation.
    """
    head_dim = config.head_dim
    half_dim = head_dim // 2  # 48
    quarter = half_dim // 2  # 24
    theta = config.rope_theta
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, half_dim, 2, dtype=torch.float32)[:quarter] / half_dim)
    )
    idx_h = torch.arange(1, grid_h + 1, dtype=torch.float32)
    idx_w = torch.arange(1, grid_w + 1, dtype=torch.float32)
    idx_ij_h = idx_h.unsqueeze(1).expand(-1, grid_w).reshape(-1)
    idx_ij_w = idx_w.unsqueeze(0).expand(grid_h, -1).reshape(-1)
    freq_h = torch.outer(idx_ij_h, inv_freq)
    freq_w = torch.outer(idx_ij_w, inv_freq)
    freq = torch.cat([freq_w, freq_h], dim=-1)  # [n_tokens, half_dim]
    return torch.cos(freq), torch.sin(freq)


def sparse_perm_and_slens(
    grid_h: int,
    grid_w: int,
    config: MuseGlimmerVisionConfig,
) -> tuple[torch.Tensor, list[int]]:
    """Return the sparse-attention permutation and block lengths."""
    gh, gw = config.pos_emb_grid_h, config.pos_emb_grid_w
    pad_h = math.ceil(grid_h / gh) * gh
    pad_w = math.ceil(grid_w / gw) * gw
    idx = torch.arange(grid_h * grid_w).view(grid_h, grid_w)
    idx = F.pad(idx, (0, pad_w - grid_w, 0, pad_h - grid_h), value=-1).flatten()
    idx = idx.view(pad_h // gh, gh, pad_w // gw, gw)
    idx = idx.permute(0, 2, 1, 3).reshape(-1)
    sp_perm = idx[idx != -1]
    valid = (idx != -1).view(-1, gh * gw)
    sp_slens = valid.sum(dim=1).tolist()
    return sp_perm.to(torch.int64), sp_slens


def block_diag_mask(slens: list[int]) -> torch.Tensor:
    """Bool block-diagonal mask ``[total, total]`` (True = attend within block)."""
    total = sum(slens)
    mask = torch.zeros(total, total, dtype=torch.bool)
    offset = 0
    for s in slens:
        mask[offset : offset + s, offset : offset + s] = True
        offset += s
    return mask


def pixel_shuffle_perm(
    grid_h: int,
    grid_w: int,
    config: MuseGlimmerVisionConfig,
) -> torch.Tensor:
    """Return the token permutation for pixel-shuffle downsampling."""
    f = config.downsample_factor
    ds_perm = torch.arange(grid_h * grid_w)
    ds_perm = ds_perm.view(grid_h // f, f, grid_w // f, f)
    ds_perm = ds_perm.permute(0, 2, 1, 3).reshape(-1)
    return ds_perm.to(torch.int64)


# Top-level bundle


@dataclass
class VisionEncoderInputs:
    """The 9 tensor inputs to the exported ``vision_encoder`` graph."""

    patches: torch.Tensor  # [1, P, patch_dim] f32
    pos_emb: torch.Tensor  # [1, P, latent] bf16
    cos_2d: torch.Tensor  # [P, head_dim//2] f32
    sin_2d: torch.Tensor  # [P, head_dim//2] f32
    sparse_perm: torch.Tensor  # [P] i64
    inv_perm: torch.Tensor  # [P] i64
    global_mask: torch.Tensor  # [1, 1, P, P] bool
    sparse_mask: torch.Tensor  # [1, 1, P, P] bool
    pixel_perm: torch.Tensor  # [P] i64

    def as_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.patches,
            self.pos_emb,
            self.cos_2d,
            self.sin_2d,
            self.sparse_perm,
            self.inv_perm,
            self.global_mask,
            self.sparse_mask,
            self.pixel_perm,
        )


def precompute_vision_inputs(
    image: torch.Tensor,  # [3, H, W] normalized to [-1, 1]
    pos_embed_table: torch.Tensor,  # [pos_grid_h*pos_grid_w, latent] bf16
    config: MuseGlimmerVisionConfig | None = None,
) -> VisionEncoderInputs:
    """Produce all host-precomputed inputs for the vision encoder from one image.

    ``image`` must already be resized so H and W are multiples of
    ``patch_size * downsample_factor`` (28) and normalized to [-1, 1] (mean/std
    0.5). Returns a :class:`VisionEncoderInputs` bundle.
    """
    if config is None:
        config = MuseGlimmerVisionConfig()

    patches, grid_h, grid_w = patchify_image(image, config)
    n_tokens = grid_h * grid_w

    pos_emb = interpolate_pos_emb(pos_embed_table, grid_h, grid_w, config)
    cos_2d, sin_2d = make_2d_rope(grid_h, grid_w, config)

    sparse_perm, sp_slens = sparse_perm_and_slens(grid_h, grid_w, config)
    inv_perm = torch.empty_like(sparse_perm)
    inv_perm[sparse_perm] = torch.arange(n_tokens, dtype=torch.int64)

    # Single image: global layers attend over all tokens (all-True); sparse
    # layers attend within each 32x32 tile (block-diagonal over sp_slens, in the
    # PERMUTED order).
    global_mask = torch.ones(1, 1, n_tokens, n_tokens, dtype=torch.bool)
    sparse_mask = block_diag_mask(sp_slens).unsqueeze(0).unsqueeze(0).contiguous()

    pixel_perm = pixel_shuffle_perm(grid_h, grid_w, config)

    return VisionEncoderInputs(
        patches=patches,
        pos_emb=pos_emb,
        cos_2d=cos_2d,
        sin_2d=sin_2d,
        sparse_perm=sparse_perm,
        inv_perm=inv_perm,
        global_mask=global_mask,
        sparse_mask=sparse_mask,
        pixel_perm=pixel_perm,
    )


def compute_grid_size(
    img_w: int,
    img_h: int,
    config: MuseGlimmerVisionConfig | None = None,
    max_tokens: int = 4096,
) -> tuple[int, int, int]:
    """Choose the closest-aspect patch grid under the image-token limit."""
    if config is None:
        config = MuseGlimmerVisionConfig()
    ph = config.patch_size * config.downsample_factor  # 28
    i_nph = img_h / ph
    i_npw = img_w / ph
    ratio = i_npw / i_nph if i_nph > 0 else 1.0
    if i_nph * i_npw > max_tokens:
        i_nph = (max_tokens / ratio) ** 0.5
        i_npw = i_nph * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(i_nph), math.ceil(i_nph)],
                [math.floor(i_npw), math.ceil(i_npw)],
            )
        )
    )
    candidates = [
        (nph, npw)
        for nph, npw in candidates
        if nph >= 1 and npw >= 1 and nph * npw <= max_tokens
    ]
    if not candidates:
        candidates = [(max(1, round(i_nph)), max(1, round(i_npw)))]
    nph, npw = min(candidates, key=lambda c: abs(c[0] / c[1] - img_h / img_w))
    return nph * ph, npw * ph, nph * npw
