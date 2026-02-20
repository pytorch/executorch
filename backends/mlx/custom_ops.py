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
"""

from typing import Optional

import torch
from torch import Tensor


# =============================================================================
# kv_cache_update: Functional KV cache update for BHSD layout
# =============================================================================


@torch.library.custom_op("mlx::kv_cache_update", mutates_args=("cache",))
def kv_cache_update(
    cache: Tensor,  # [B, H, S_max, D] - mutated in place
    new_values: Tensor,  # [B, H, S, D]
    start_pos: int,
) -> Tensor:
    """
    Mutating KV cache update that modifies cache in place.

    This op updates the cache at positions [start_pos, start_pos + S) with
    new_values. The cache is mutated in place, similar to llama.update_cache.

    Args:
        cache: Cache tensor of shape [B, H, S_max, D] (BHSD layout) - mutated
        new_values: New values to insert of shape [B, H, S, D]
        start_pos: Starting position index for insertion

    Returns:
        A dummy tensor (1,) - the return value is not semantically meaningful
        but is required for slot management during export. This follows the
        same pattern as llama.update_cache.

    Note:
        The BHSD layout matches what torch SDPA expects, avoiding transposition.
    """
    seq_len = new_values.size(2)
    end_pos = start_pos + seq_len

    # Validate bounds
    assert end_pos <= cache.size(2), (
        f"Cache update out of bounds: start_pos={start_pos}, "
        f"seq_len={seq_len}, cache_len={cache.size(2)}"
    )

    # Mutate cache in place
    cache[:, :, start_pos:end_pos, :] = new_values

    # Return dummy tensor like llama.update_cache does
    return torch.empty((1,), dtype=new_values.dtype, device=new_values.device)


@torch.library.register_fake("mlx::kv_cache_update")
def kv_cache_update_fake(
    cache: Tensor,
    new_values: Tensor,
    start_pos: int,
) -> Tensor:
    """Fake implementation for tracing - returns dummy tensor like llama.update_cache."""
    return torch.empty((1,), dtype=new_values.dtype, device="meta")


# =============================================================================
# custom_sdpa: Scaled Dot-Product Attention with KV cache slicing
# =============================================================================


@torch.library.custom_op("mlx::custom_sdpa", mutates_args=())
def mlx_custom_sdpa(
    query: Tensor,  # [B, num_heads, seq_len, head_dim] - BHSD
    key: Tensor,  # [B, num_kv_heads, kv_len, head_dim] - BHSD (FULL cache)
    value: Tensor,  # [B, num_kv_heads, kv_len, head_dim] - BHSD (FULL cache)
    start_pos: int,  # FIRST position in current batch (0-indexed)
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    MLX custom SDPA with K/V cache slicing.

    This op uses BHSD layout (matching PyTorch SDPA and MLX's SdpaNode).
    It receives the FULL K/V cache and slices to [0:stop_pos] before computing
    attention, where stop_pos = start_pos + query_seq_len.

    The semantics follow executorch's llama.custom_sdpa:
    - start_pos: FIRST position of the current query batch
    - For prefill with 7 tokens at positions [0,1,2,3,4,5,6]: start_pos=0, stop_pos=7
    - For decode at position 10: start_pos=10, stop_pos=11

    Args:
        query: Query tensor [B, num_heads, seq_len, head_dim]
        key: Key cache [B, num_kv_heads, kv_len, head_dim] - FULL cache
        value: Value cache [B, num_kv_heads, kv_len, head_dim] - FULL cache
        start_pos: FIRST position in current batch (SymInt)
        attn_mask: Optional attention mask (only used when is_causal=False)
        dropout_p: Dropout probability (default 0.0)
        is_causal: Whether to apply causal masking (default False)
        scale: Attention scale factor (default 1/sqrt(head_dim))

    Returns:
        Attention output [B, num_heads, seq_len, head_dim] - BHSD
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5

    # Compute stop_pos = start_pos + query_seq_len
    # BHSD layout: seq_len is at dim 2
    query_seq_len = query.shape[2]
    stop_pos = start_pos + query_seq_len

    # Constrain symbolic shapes so torch.export can resolve guards.
    # start_pos is data-dependent (from input_pos), so the slice
    # stop_pos > kv_len comparison is unresolvable without these hints.
    torch._check_is_size(start_pos)
    torch._check(stop_pos <= key.shape[2])

    # Slice K/V to valid cache entries [0:stop_pos]
    key_sliced = key[:, :, :stop_pos, :]
    value_sliced = value[:, :, :stop_pos, :]

    # Handle GQA: expand K/V heads to match query heads
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        key_sliced = key_sliced.repeat_interleave(num_groups, dim=1)
        value_sliced = value_sliced.repeat_interleave(num_groups, dim=1)

    # Build explicit lower-right aligned causal mask to match MLX's SdpaNode.
    # PyTorch's is_causal=True uses upper-left alignment when Q_len != K_len,
    # but for KV-cache inference q[i] is at context position (start_pos + i)
    # and should attend to all positions 0..start_pos+i (lower-right).
    if is_causal:
        L, S = query.shape[2], key_sliced.shape[2]
        offset = S - L  # equals start_pos
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=offset
        )
        attn_mask = torch.where(mask, 0.0, float("-inf")).to(query.dtype)

    # Compute SDPA - returns BHSD
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key_sliced,
        value_sliced,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
    )


@torch.library.register_fake("mlx::custom_sdpa")
def mlx_custom_sdpa_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    start_pos: int,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """Fake implementation for tracing - returns BHSD shape (same as query)."""
    return query.new_empty(query.shape)


# =============================================================================
# rope: Rotary Position Embedding (single tensor)
# =============================================================================


@torch.library.custom_op("mlx::rope", mutates_args=())
def rope(
    x: Tensor,  # (B, H, T, D)
    dims: int,
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
        dims: Number of feature dimensions to rotate. If less than D,
            only the first `dims` dimensions are rotated and the rest
            are left unchanged.
        pos: Starting position index (int, not tensor)
        traditional: Whether to use traditional RoPE formulation
        base: Base for frequency computation
        scale: Scale factor for frequencies
        freqs: Optional precomputed frequencies

    Returns:
        Rotated tensor of the same shape
    """
    Dh = int(dims)

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

    # Split into rotated and unrotated portions
    x_rot = x[..., :Dh]
    x_pass = x[..., Dh:]

    if traditional:
        # Interleaved pairs: (x[0],x[1]), (x[2],x[3]), ...
        x1 = x_rot[..., 0::2]  # even indices
        x2 = x_rot[..., 1::2]  # odd indices
        xr = x1 * cos - x2 * sin
        xi = x1 * sin + x2 * cos
        rotated = torch.stack([xr, xi], dim=-1).flatten(-2)
    else:
        # Split-half: first half paired with second half
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        xr = x1 * cos - x2 * sin
        xi = x1 * sin + x2 * cos
        rotated = torch.cat([xr, xi], dim=-1)

    if x_pass.shape[-1] > 0:
        return torch.cat([rotated, x_pass], dim=-1)
    return rotated


@torch.library.register_fake("mlx::rope")
def rope_fake(
    x: Tensor,
    dims: int,
    pos: int,
    traditional: bool = False,
    base: float = 500000.0,
    scale: float = 1.0,
    freqs: Optional[Tensor] = None,
) -> Tensor:
    """Fake implementation for tracing."""
    return x.new_empty(x.shape)
