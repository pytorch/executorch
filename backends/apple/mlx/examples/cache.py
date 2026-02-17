#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared KV cache utilities for MLX delegate examples.

Provides reusable KV cache implementations optimized for the MLX backend:

- ETKVCache: Single-layer cache with BHSD layout, ExecutorTorch llama KVCache interface
- HFStaticCache: Multi-layer cache following HuggingFace's StaticCache interface

Usage with HuggingFace models:
    from executorch.backends.apple.mlx.examples.source_transformation import (
        replace_hf_cache_with_mlx,
    )

    # Load HF model with StaticCache
    model = AutoModelForCausalLM.from_pretrained(...)
    model = model.to_exportable(max_batch_size=1, max_cache_len=4096)

    # Replace with MLX-optimized cache before export
    replace_hf_cache_with_mlx(model, model.config, max_cache_len=4096)
"""

from typing import Tuple

import torch
import torch.nn as nn

# Import MLX custom ops to register mlx::kv_cache_update
from executorch.backends.apple.mlx import custom_ops as _mlx_custom_ops  # noqa: F401


class ETKVCache(nn.Module):
    """
    MLX-optimized KV cache with ExecutorTorch llama KVCache interface.

    This class follows the same interface as examples/models/llama/attention.py KVCache,
    making it a drop-in replacement, but uses the mlx::kv_cache_update op internally
    which is optimized for the MLX delegate.

    The cache uses BHSD layout [B, H, S, D] which matches what torch SDPA expects.

    The ``update`` method accepts ``input_pos`` as either a ``torch.Tensor`` or a
    plain ``int`` / SymInt.  When a tensor is passed, ``item()`` is called internally
    to extract the start position, which introduces an unbacked SymInt during
    ``torch.export``.  Extracting a SymInt has a cost because it creates a new
    symbolic variable and associated constraints in the exported program.  In a
    multi-layer model, prefer extracting the SymInt once and passing the resulting
    int/SymInt to every layer's ``update`` call rather than passing the tensor
    repeatedly:

    .. code-block:: python

        # Preferred: extract once, pass to all layers
        start_pos = input_pos[0].item()
        for layer_cache in caches:
            layer_cache.update(start_pos, k_val, v_val)

        # Avoid: each layer re-extracts from the tensor
        for layer_cache in caches:
            layer_cache.update(input_pos, k_val, v_val)

    Example:
        >>> cache = ETKVCache(
        ...     max_batch_size=1,
        ...     max_context_length=4096,
        ...     n_heads=32,
        ...     head_dim=128,
        ...     enable_dynamic_shape=True,
        ... )
        >>> # With tensor input_pos
        >>> input_pos = torch.tensor([0])
        >>> k_val = torch.randn(1, 32, 10, 128)  # [B, H, S, D]
        >>> v_val = torch.randn(1, 32, 10, 128)  # [B, H, S, D]
        >>> k_cache, v_cache = cache.update(input_pos, k_val, v_val)
        >>>
        >>> # With int/SymInt input_pos (preferred in multi-layer loops)
        >>> start_pos = input_pos[0].item()
        >>> k_cache, v_cache = cache.update(start_pos, k_val, v_val)
    """

    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        enable_dynamic_shape: bool,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize KV cache buffers.

        Args:
            max_batch_size: Maximum batch size
            max_context_length: Maximum sequence length the cache can hold
            n_heads: Number of attention heads (key/value heads for GQA)
            head_dim: Dimension per head
            enable_dynamic_shape: Whether dynamic shapes are enabled (kept for interface
                                  compatibility, but MLX always uses dynamic-style update)
            dtype: Data type for cache buffers
        """
        super().__init__()
        assert (
            max_batch_size == 1
        ), f"Only max_batch_size=1 is supported, but got {max_batch_size}"
        self.max_batch_size = max_batch_size
        self.max_context_length = max_context_length
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.enable_dynamic_shape = enable_dynamic_shape

        # Initialize cache buffers [B, H, T_max, D] - BHSD layout
        cache_shape = (max_batch_size, n_heads, max_context_length, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )

    def update(
        self, input_pos: torch.Tensor | int, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K/V states and return FULL cache.

        This method follows the same signature as examples/models/llama/attention.py KVCache.

        Args:
            input_pos: Start position — either a position tensor [S] or an int/SymInt
            k_val: New key states [B, H, S, D]
            v_val: New value states [B, H, S, D]

        Returns:
            Tuple of (k_cache, v_cache) - slices of the FULL cache buffers
        """
        # Extract start position as int (SymInt during tracing)
        if isinstance(input_pos, torch.Tensor):
            start_pos = input_pos[0].item()
            seq_len = k_val.size(2)
            torch._check(seq_len == v_val.size(2))
            torch._check_is_size(start_pos)
            torch._check(start_pos + seq_len <= self.max_context_length)
        else:
            start_pos = input_pos

        # Use MLX custom op for cache update (mutates in place)
        torch.ops.mlx.kv_cache_update(self.k_cache, k_val, start_pos)
        torch.ops.mlx.kv_cache_update(self.v_cache, v_val, start_pos)

        # Return full slices of the cache (creates new tensor nodes in the graph)
        # This avoids the issue where the same tensor is both BUFFER_MUTATION and USER_OUTPUT
        return self.k_cache[:, :, :, :], self.v_cache[:, :, :, :]


# =============================================================================
# HFStaticCache - Standalone HuggingFace-compatible Static Cache
# =============================================================================

from transformers.cache_utils import StaticCache


class HFStaticCache(StaticCache):
    """
    MLX-optimized Static KV Cache that follows HuggingFace's StaticCache interface.

    This cache is designed to be a drop-in replacement for HuggingFace's StaticCache
    when exporting models for the MLX backend. It uses mlx::kv_cache_update internally
    which is optimized for the MLX delegate.

    The cache supports multi-layer models by maintaining separate K/V buffers per layer,
    matching the HF StaticCache behavior where `update()` takes a `layer_idx` argument.

    Layout: BHSD [batch_size, num_heads, max_cache_len, head_dim]

    Example:
        >>> from transformers import AutoConfig
        >>> config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> cache = HFStaticCache(config, max_batch_size=1, max_cache_len=4096)
        >>> # In attention layer:
        >>> k_out, v_out = cache.update(k_states, v_states, layer_idx=0,
        ...                              cache_kwargs={"cache_position": pos_tensor})
    """

    def __init__(
        self,
        config,
        max_batch_size: int = 1,
        max_cache_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize MLX Static Cache.

        Args:
            config: HuggingFace model config with num_hidden_layers, num_key_value_heads,
                   num_attention_heads, hidden_size, and optionally head_dim
            max_batch_size: Maximum batch size (default: 1)
            max_cache_len: Maximum cache length. If None, uses config.max_position_embeddings
            device: Device for cache tensors (default: None = CPU)
            dtype: Data type for cache tensors (default: torch.float32)
        """
        # Resolve dimensions from config BEFORE calling parent
        num_layers = config.num_hidden_layers
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        actual_max_cache_len = max_cache_len or getattr(
            config, "max_position_embeddings", 2048
        )

        # Initialize parent StaticCache with required arguments
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=actual_max_cache_len,
            device=device,
            dtype=dtype,
        )
        # Call early_initialization to ensure parent's layers are fully initialized
        self.early_initialization(
            batch_size=max_batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        # Store dimensions as instance attributes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Create ETKVCache wrappers for each layer - these use mlx::kv_cache_update
        # Named 'kv_cache' to match optimum-executorch's ETCustomStaticCache pattern
        self.kv_cache = nn.ModuleList(
            [
                ETKVCache(
                    max_batch_size=max_batch_size,
                    max_context_length=actual_max_cache_len,
                    n_heads=num_heads,
                    head_dim=head_dim,
                    enable_dynamic_shape=True,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Move to device if specified
        if device is not None:
            self.to(device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key/value states for a specific layer.

        This method follows HuggingFace's StaticCache.update() signature.

        Args:
            key_states: New key states [batch_size, num_heads, seq_len, head_dim]
            value_states: New value states [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Index of the layer to update
            cache_kwargs: Dictionary containing 'cache_position' tensor with start position

        Returns:
            Tuple of (key_cache, value_cache) for the full cache after update
        """
        assert (
            cache_kwargs is not None
        ), "cache_kwargs must be provided with 'cache_position'"
        cache_position = cache_kwargs.get("cache_position")
        assert (
            cache_position is not None
        ), "cache_position must be provided in cache_kwargs"
        assert isinstance(
            cache_position, torch.Tensor
        ), "cache_position must be a tensor"

        # Pass cache_position tensor directly to ETKVCache.update()
        # ETKVCache extracts start_pos internally via input_pos[0].item()
        return self.kv_cache[layer_idx].update(cache_position, key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Get the current sequence length in the cache.

        Note: This is approximate - returns the number of non-zero positions.

        Args:
            layer_idx: Layer index to check (default: 0)

        Returns:
            Approximate sequence length
        """
        # Check how many positions have been filled by looking for non-zero values
        k_cache = self.kv_cache[layer_idx].k_cache
        # Check if any value in the head_dim is non-zero for each position
        return (k_cache[0, 0, :, 0] != 0).sum().item()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Get the maximum cache length."""
        return self.max_cache_len

    def reset(self):
        """Reset all cache buffers to zero."""
        for layer_cache in self.kv_cache:
            layer_cache.k_cache.zero_()
            layer_cache.v_cache.zero_()
