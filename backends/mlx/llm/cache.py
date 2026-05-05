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
"""

import inspect
from typing import Tuple

import torch
import torch.nn as nn

# Import MLX custom ops to register mlx::kv_cache_update
from executorch.backends.mlx import custom_ops as _mlx_custom_ops  # noqa: F401


def resolve_hf_text_config(config):
    """Return the text config for multimodal HF models, or the config itself."""
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


def resolve_hf_cache_layout(config):
    """
    Return per-cache-layer metadata for HuggingFace hybrid/static caches.

    Some models such as Gemma 4 use different KV geometries depending on the
    attention layer type. Match the upstream `transformers` hybrid cache layout
    so our replacement cache allocates the same number of layers with the same
    `(num_heads, head_dim)` for each backing cache entry.
    """
    text_config = resolve_hf_text_config(config)
    layer_types = getattr(text_config, "layer_types", None)

    if layer_types is None:
        if getattr(text_config, "sliding_window", None) is not None:
            layer_types = [
                "sliding_attention" for _ in range(text_config.num_hidden_layers)
            ]
        else:
            layer_types = [
                "full_attention" for _ in range(text_config.num_hidden_layers)
            ]
    else:
        layer_types = list(layer_types)

    if hasattr(text_config, "num_kv_shared_layers"):
        layer_types = layer_types[: -text_config.num_kv_shared_layers]

    if hasattr(text_config, "global_head_dim"):
        head_dims = [
            (
                text_config.global_head_dim
                if layer_type == "full_attention"
                else text_config.head_dim
            )
            for layer_type in layer_types
        ]
        num_heads = [
            (
                text_config.num_global_key_value_heads
                if layer_type == "full_attention"
                and getattr(text_config, "attention_k_eq_v", False)
                else text_config.num_key_value_heads
            )
            for layer_type in layer_types
        ]
    else:
        head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        num_head = getattr(
            text_config, "num_key_value_heads", text_config.num_attention_heads
        )
        head_dims = [head_dim for _ in layer_types]
        num_heads = [num_head for _ in layer_types]

    return layer_types, num_heads, head_dims


class KVCache(nn.Module):
    """
    MLX-optimized KV cache with ExecuTorch llama KVCache interface.

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
        >>> cache = KVCache(
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

        if isinstance(input_pos, torch.Tensor):
            start_pos = input_pos[0].item()
            seq_len = k_val.size(2)
            torch._check(seq_len == v_val.size(2))
            torch._check(start_pos >= 0)
            torch._check(start_pos + seq_len <= self.max_context_length)
        else:
            start_pos = input_pos

        torch.ops.mlx.kv_cache_update(self.k_cache, k_val, start_pos)
        torch.ops.mlx.kv_cache_update(self.v_cache, v_val, start_pos)

        # Return full slices of the cache (creates new tensor nodes in the graph)
        # This avoids the issue where the same tensor is both BUFFER_MUTATION and USER_OUTPUT
        return self.k_cache[:, :, :, :], self.v_cache[:, :, :, :]


class RingBufferKVCache(nn.Module):
    """
    Ring buffer KV cache for sliding window attention.

    Instead of a linear cache that fills up and stops, this cache wraps around:
    write_pos = start_pos % window_size. When the cache is full, new tokens
    overwrite the oldest ones, enabling infinite-length generation.

    The attention mask is computed branchlessly from ``start_pos`` and
    ``window_size`` alone using ``torch.where`` — no mutable position-tracking
    buffers and no Python if/else that would create torch.export guards.

    Mask creation is NOT done here — following optimum-executorch's pattern,
    the attention function creates the mask lazily by accessing the cache
    via a closure. This avoids tracing issues with torch.export.

    Layout: BHSD [batch_size, num_heads, window_size, head_dim]

    Example:
        >>> cache = RingBufferKVCache(
        ...     max_batch_size=1,
        ...     max_context_length=512,
        ...     n_heads=4,
        ...     head_dim=256,
        ...     dtype=torch.bfloat16,
        ... )
        >>> k_val = torch.randn(1, 4, 1, 256)
        >>> v_val = torch.randn(1, 4, 1, 256)
        >>> k_cache, v_cache = cache.update(start_pos=0, k_val=k_val, v_val=v_val)
    """

    def __init__(
        self,
        max_batch_size: int,
        max_context_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert (
            max_batch_size == 1
        ), f"Only max_batch_size=1 is supported, but got {max_batch_size}"
        self.max_batch_size = max_batch_size
        self.max_context_length = max_context_length
        self.window_size = max_context_length
        self.buffer_size = 2 * max_context_length
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Cache buffers [B, H, 2*window_size, D]
        # 2× buffer ensures multi-token writes never overwrite data that
        # earlier queries in the same batch still need (matches ET behavior).
        cache_shape = (max_batch_size, n_heads, self.buffer_size, head_dim)
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
        Update cache with new K/V states using ring buffer semantics.

        Args:
            input_pos: Start position — either a position tensor [S] or an int/SymInt
            k_val: New key states [B, H, S, D]
            v_val: New value states [B, H, S, D]

        Returns:
            Tuple of (k_cache, v_cache) — full ring buffer slices
        """
        if isinstance(input_pos, torch.Tensor):
            start_pos = input_pos[0].item()
            seq_len = k_val.size(2)
            torch._check(seq_len == v_val.size(2))
            torch._check(start_pos >= 0)
            torch._check(seq_len <= self.window_size)
        else:
            start_pos = input_pos

        torch.ops.mlx.kv_cache_update(
            self.k_cache, k_val, start_pos, ring_size=self.buffer_size
        )
        torch.ops.mlx.kv_cache_update(
            self.v_cache, v_val, start_pos, ring_size=self.buffer_size
        )

        return self.k_cache[:, :, :, :], self.v_cache[:, :, :, :]

    def create_sliding_window_mask(self, start_pos: int, seq_len: int) -> torch.Tensor:
        """
        Build attention mask for the ring buffer — branchless, no mutable state.

        Reconstructs the slot→position mapping from ``start_pos`` and
        ``buffer_size`` alone using ``torch.where``, avoiding both Python
        if/else (which creates torch.export guards) and mutable position-
        tracking buffers (which require extra kv_cache_update calls and
        complicate partitioning).

        Returns:
            Additive mask [1, 1, seq_len, buffer_size] in the cache's dtype,
            where 0 = attend, -inf = block.
        """
        w = self.window_size
        b = self.buffer_size
        end_pos = start_pos + seq_len

        # Slot indices [buffer_size]
        slots = torch.arange(b, dtype=torch.long)

        last_write_slot = (end_pos - 1) % b
        current_cycle_base = end_pos - 1 - last_write_slot
        pos_current = current_cycle_base + slots
        pos_previous = current_cycle_base - b + slots

        cache_pos = torch.where(slots <= last_write_slot, pos_current, pos_previous)

        # Query positions [seq_len, 1]
        pos_q = (start_pos + torch.arange(seq_len, dtype=torch.long)).view(-1, 1)

        # Delta from query to each cached position [seq_len, buffer_size]
        delta = pos_q - cache_pos

        # A slot is attendable if: filled (pos >= 0), causal (delta >= 0),
        # and within the sliding window (delta < w)
        attn_mask = (cache_pos >= 0) & (delta >= 0) & (delta < w)

        # Use cache dtype (e.g. bf16) to avoid float32 AsTypeNode casts in SDPA
        dtype = self.k_cache.dtype
        zero = torch.zeros(1, dtype=dtype)
        neg_inf = torch.full((1,), float("-inf"), dtype=dtype)
        return torch.where(attn_mask, zero, neg_inf).unsqueeze(0).unsqueeze(0)


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
        # Resolve dimensions from the text config before calling parent. Multimodal
        # configs like Gemma 4 expose transformer dims under text_config.
        text_config = resolve_hf_text_config(config)
        layer_types, num_heads, head_dims = resolve_hf_cache_layout(config)
        num_model_layers = text_config.num_hidden_layers
        actual_max_cache_len = max_cache_len or getattr(
            text_config, "max_position_embeddings", 2048
        )

        # Initialize parent StaticCache with required arguments
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=actual_max_cache_len,
            device=device,
            dtype=dtype,
        )
        # Newer HF cache implementations already support per-layer layouts in
        # early_initialization(). Keep that path for Gemma 4, and only fall
        # back to manual layer initialization for the older CI-pinned API.
        try:
            self.early_initialization(
                batch_size=max_batch_size,
                num_heads=num_heads,
                head_dim=head_dims,
                dtype=dtype,
                device=device,
            )
        except TypeError:
            for layer, layer_num_heads, layer_head_dim in zip(
                self.layers, num_heads, head_dims
            ):
                fake_keys_tensor = torch.zeros(
                    (max_batch_size, layer_num_heads, 0, layer_head_dim),
                    dtype=dtype,
                    device=device,
                )
                lazy_init_sig = inspect.signature(layer.lazy_initialization)
                # Older pinned HF caches take a single fake tensor, while newer
                # versions expect both key_states and value_states separately.
                if len(lazy_init_sig.parameters) == 1:
                    layer.lazy_initialization(fake_keys_tensor)
                else:
                    fake_values_tensor = torch.zeros(
                        (max_batch_size, layer_num_heads, 0, layer_head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    layer.lazy_initialization(fake_keys_tensor, fake_values_tensor)

        # Some models (for example Gemma 4) only allocate cache entries for the
        # non-shared KV layers. Mirror the parent StaticCache layout exactly so
        # layer_idx values passed to update() line up with our backing cache.
        num_cache_layers = len(self.layers)

        # Store dimensions as instance attributes
        self.num_model_layers = num_model_layers
        self.num_layers = num_cache_layers
        self.layer_types = layer_types
        self.num_heads = num_heads
        self.head_dim = head_dims

        # Create KVCache wrappers for each layer - these use mlx::kv_cache_update
        # Named 'kv_cache' to match optimum-executorch's ETCustomStaticCache pattern
        self.kv_cache = nn.ModuleList(
            [
                KVCache(
                    max_batch_size=max_batch_size,
                    max_context_length=actual_max_cache_len,
                    n_heads=layer_num_heads,
                    head_dim=layer_head_dim,
                    enable_dynamic_shape=True,
                    dtype=dtype,
                )
                for layer_num_heads, layer_head_dim in zip(num_heads, head_dims)
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
            cache_kwargs: Optional dictionary containing 'cache_position' tensor
                with start position. Newer HF StaticCache callers seed
                `self.layers[layer_idx].cumulative_length` directly and do not
                pass cache_kwargs.

        Returns:
            Tuple of (key_cache, value_cache) for the full cache after update
        """
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position")
        else:
            cache_position = None

        if cache_position is None:
            # Current HF ExecuTorch wrappers copy the requested cache position
            # into each StaticCache layer's cumulative_length before forward().
            if hasattr(self.layers[layer_idx], "cumulative_length"):
                cache_position = self.layers[layer_idx].cumulative_length
            else:
                raise RuntimeError(
                    "cache_position was not provided and the pinned "
                    "transformers StaticCache layer does not expose "
                    "cumulative_length"
                )

        assert isinstance(
            cache_position, torch.Tensor
        ), "cache_position must be a tensor"

        # Pass cache_position tensor directly to KVCache.update()
        # KVCache extracts start_pos internally via input_pos[0].item()
        return self.kv_cache[layer_idx].update(cache_position, key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Approximate sequence length (counts non-zero cache positions)."""
        k_cache = self.kv_cache[layer_idx].k_cache
        # Check if any value in the head_dim is non-zero for each position
        return (k_cache[0, 0, :, 0] != 0).sum().item()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.max_cache_len

    def reset(self):
        for layer_cache in self.kv_cache:
            layer_cache.k_cache.zero_()
            layer_cache.v_cache.zero_()
