#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared KV cache utilities for MLX delegate examples.

Provides a reusable KV cache update mechanism using torch.ops.llama.update_cache
that works consistently across different model architectures (Llama, Whisper, etc.).

Supports two cache implementations:
- MLXKVCache: Uses [B, H, S, D] layout (default)
- CustomKVCache: Uses [B, S, H, D] layout (from examples/models/llama)
"""

from typing import Tuple

import torch
import torch.nn as nn


def kv_update(
    k_cache: torch.Tensor,  # [B, H, T_max, D]
    v_cache: torch.Tensor,  # [B, H, T_max, D]
    k_step: torch.Tensor,  # [B, H, T_step, D]
    v_step: torch.Tensor,  # [B, H, T_step, D]
    input_pos: int,  # Position as int (SymInt during tracing from .item())
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update KV cache using llama.update_cache and return FULL cache.

    This function provides a consistent pattern for updating KV caches across
    different model architectures. It uses torch.ops.llama.update_cache which is
    well-tested for MLX export.

    Note: This function does NOT window the cache. The caller is responsible for
    windowing if needed (e.g., k_cache[:, :, :end_pos, :]).

    Args:
        k_cache: Key cache buffer [B, H, T_max, D]
        v_cache: Value cache buffer [B, H, T_max, D]
        k_step: New key states to insert [B, H, T_step, D]
        v_step: New value states to insert [B, H, T_step, D]
        input_pos: Starting position for insertion (as int/SymInt)

    Returns:
        Tuple of (k_cache, v_cache) - the FULL cache buffers after update

    Note:
        The function transposes between [B, H, S, D] and [B, S, H, D] layouts because
        update_cache expects [B, S, H, D] format, while SDPA uses [B, H, S, D] format.
    """
    # Transpose cache and inputs from [B, H, S, D] to [B, S, H, D] for update_cache
    k_cache_view = k_cache.transpose(1, 2)
    v_cache_view = v_cache.transpose(1, 2)
    k_step_t = k_step.transpose(1, 2)
    v_step_t = v_step.transpose(1, 2)

    # Use llama.update_cache (well-tested pattern for MLX)
    torch.ops.llama.update_cache(k_step_t, k_cache_view, input_pos)
    torch.ops.llama.update_cache(v_step_t, v_cache_view, input_pos)

    # Return FULL cache - no windowing
    return k_cache, v_cache


class KVCache(nn.Module):
    """
    Reusable KV cache module for attention mechanisms.

    This module manages key and value cache buffers and provides a clean
    callable interface for updating and windowing the cache. It's designed
    to work with different attention patterns (self-attention, cross-attention, etc.).

    Example:
        >>> cache = KVCache(num_heads=32, head_dim=128, max_cache_len=4096)
        >>> k_new = torch.randn(1, 32, 10, 128)  # New keys
        >>> v_new = torch.randn(1, 32, 10, 128)  # New values
        >>> k_win, v_win = cache(k_new, v_new, pos=0)  # Callable interface
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_cache_len: int,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize KV cache buffers.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension per head
            max_cache_len: Maximum sequence length the cache can hold
            dtype: Data type for cache buffers
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len

        # Initialize cache buffers [B, H, T_max, D]
        k0 = torch.zeros((1, num_heads, max_cache_len, head_dim), dtype=dtype)
        v0 = torch.zeros((1, num_heads, max_cache_len, head_dim), dtype=dtype)
        self.register_buffer("k_cache", k0, persistent=False)
        self.register_buffer("v_cache", v0, persistent=False)

    def forward(
        self,
        k_step: torch.Tensor,  # [B, H, T_step, D]
        v_step: torch.Tensor,  # [B, H, T_step, D]
        input_pos: int,  # Position as int (SymInt during tracing)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K/V states and return FULL cache.

        This method makes KVCache callable: `k_cache, v_cache = cache(k, v, pos)`

        Note: Returns the FULL cache buffer. The caller is responsible for windowing
        if needed (e.g., k_cache[:, :, :end_pos, :]).

        Args:
            k_step: New key states [B, H, T_step, D]
            v_step: New value states [B, H, T_step, D]
            input_pos: Starting position for insertion

        Returns:
            Tuple of (k_cache, v_cache) - FULL cache buffers after update
        """
        return kv_update(self.k_cache, self.v_cache, k_step, v_step, input_pos)
