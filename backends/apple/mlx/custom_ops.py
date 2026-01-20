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
- apply_rope: Rotary Position Embedding application
"""

from typing import Optional, Tuple

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
# apply_rope: Rotary Position Embedding
# =============================================================================


@torch.library.custom_op("mlx::apply_rope", mutates_args=())
def apply_rope(
    q_in: Tensor,  # (B, Hq, T, D)
    k_in: Tensor,  # (B, Hk, T, D)
    head_dim: int,
    pos: int,  # int, not tensor
    traditional: bool = False,
    base: float = 500000.0,
    scale: float = 1.0,
    freqs: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Apply Rotary Position Embedding to query and key tensors.

    Args:
        q_in: Query tensor of shape (B, Hq, T, D)
        k_in: Key tensor of shape (B, Hk, T, D)
        head_dim: Dimension of each attention head
        pos: Starting position index (int, not tensor)
        traditional: Whether to use traditional RoPE formulation
        base: Base for frequency computation
        scale: Scale factor for frequencies
        freqs: Optional precomputed frequencies

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    Dh = int(head_dim)
    assert q_in.size(-1) == Dh and k_in.size(-1) == Dh, "head_dim mismatch"

    # unpack as (B, H, T, D)
    B, Hq, T, _ = q_in.shape
    B2, Hk, T2, _ = k_in.shape
    assert B == B2 and T == T2, "RoPE expects q and k to have same B,T"
    half = Dh // 2

    if freqs is None:
        # [1, 1, 1, half] to broadcast over B,H,T
        i = torch.arange(half, device=q_in.device, dtype=torch.float32)
        inv_freq = (base ** (-2.0 * i / Dh)).view(1, 1, 1, half)

        # positions: [1, 1, T, 1]
        pos_range = torch.arange(
            pos, pos + T, device=q_in.device, dtype=torch.float32
        ).view(1, 1, T, 1)

        # final angles: [1, 1, T, half]
        angles = (pos_range * inv_freq) * float(scale)
    else:
        # assume freqs is already per-position, just reshape to [1,1,T,half]
        angles = freqs.to(torch.float32).view(1, 1, T, half)

    cos = angles.cos().to(q_in.dtype)  # [1,1,T,half]
    sin = angles.sin().to(q_in.dtype)  # [1,1,T,half]

    def rot(x: Tensor) -> Tensor:
        # x: [B, H, T, D]
        x1, x2 = x[..., :half], x[..., half : 2 * half]
        xr = x1 * cos - x2 * sin
        xi = x1 * sin + x2 * cos
        if 2 * half != Dh:
            return torch.cat([xr, xi, x[..., 2 * half :]], dim=-1)
        return torch.cat([xr, xi], dim=-1)

    q_out = rot(q_in)
    k_out = rot(k_in)
    return q_out, k_out


@torch.library.register_fake("mlx::apply_rope")
def apply_rope_fake(
    q_in: Tensor,
    k_in: Tensor,
    head_dim: int,
    pos: int,
    traditional: bool = False,
    base: float = 500000.0,
    scale: float = 1.0,
    freqs: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Fake implementation for tracing."""
    return (
        q_in.new_empty(q_in.shape),
        k_in.new_empty(k_in.shape),
    )
