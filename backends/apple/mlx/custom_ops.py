#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Custom MLX operator definitions.

This module defines custom operators that are supported by the MLX backend.
These ops are used during model export to represent operations that MLX
can execute efficiently but may not have direct PyTorch equivalents.

The ops are registered using torch.library and include:
- rms_norm: RMSNorm normalization
- rope: Rotary Position Embedding (single tensor)
"""

from typing import Optional

import torch
from torch import Tensor


# =============================================================================
# rms_norm: RMSNorm normalization
# =============================================================================


@torch.library.custom_op("mlx::rms_norm", mutates_args=())
def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """
    RMSNorm normalization.

    Args:
        x: Input tensor of shape (..., hidden_dim)
        weight: Weight tensor of shape (hidden_dim,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of the same shape as x
    """
    x_f = x.to(torch.float32)
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps)
    y = y.to(x.dtype)
    return y * weight.to(x.dtype)


@torch.library.register_fake("mlx::rms_norm")
def rms_norm_fake(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Fake implementation for tracing."""
    return x.new_empty(x.shape)


# =============================================================================
# rope: Rotary Position Embedding (single tensor)
# =============================================================================


@torch.library.custom_op("mlx::rope", mutates_args=())
def rope(
    x: Tensor,  # (B, H, T, D)
    head_dim: int,
    pos: int,  # int, not tensor
    traditional: bool = False,
    base: float = 500000.0,
    scale: float = 1.0,
    freqs: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply Rotary Position Embedding to a single tensor.

    Args:
        x: Input tensor of shape (B, H, T, D)
        head_dim: Dimension of each attention head
        pos: Starting position index (int, not tensor)
        traditional: Whether to use traditional RoPE formulation
        base: Base for frequency computation
        scale: Scale factor for frequencies
        freqs: Optional precomputed frequencies

    Returns:
        Rotated tensor of the same shape
    """
    Dh = int(head_dim)
    assert x.size(-1) == Dh, "head_dim mismatch"

    B, H, T, _ = x.shape
    half = Dh // 2

    if freqs is None:
        # [1, 1, 1, half] to broadcast over B,H,T
        i = torch.arange(half, device=x.device, dtype=torch.float32)
        inv_freq = (base ** (-2.0 * i / Dh)).view(1, 1, 1, half)

        # positions: [1, 1, T, 1]
        pos_range = torch.arange(
            pos, pos + T, device=x.device, dtype=torch.float32
        ).view(1, 1, T, 1)

        # final angles: [1, 1, T, half]
        angles = (pos_range * inv_freq) * float(scale)
    else:
        # assume freqs is already per-position, just reshape to [1,1,T,half]
        angles = freqs.to(torch.float32).view(1, 1, T, half)

    cos = angles.cos().to(x.dtype)  # [1,1,T,half]
    sin = angles.sin().to(x.dtype)  # [1,1,T,half]

    # x: [B, H, T, D]
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    xr = x1 * cos - x2 * sin
    xi = x1 * sin + x2 * cos
    if 2 * half != Dh:
        return torch.cat([xr, xi, x[..., 2 * half :]], dim=-1)
    return torch.cat([xr, xi], dim=-1)


@torch.library.register_fake("mlx::rope")
def rope_fake(
    x: Tensor,
    head_dim: int,
    pos: int,
    traditional: bool = False,
    base: float = 500000.0,
    scale: float = 1.0,
    freqs: Optional[Tensor] = None,
) -> Tensor:
    """Fake implementation for tracing."""
    return x.new_empty(x.shape)
