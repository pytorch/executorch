# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple, Union

import torch


# If transformers is not installed, raise an ImportError
try:
    from transformers.cache_utils import HybridCache, StaticCache
except ImportError:
    raise ImportError(
        "transformers is not installed. Please install it to use Static/HybridCache."
    )

try:
    from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
        CustomKVCache,
        CustomRingKVCache,
    )
except ImportError:
    raise ImportError(
        "ExecutorTorch is not installed. Please install it to use Custom Cache."
    )


class ETCustomStaticCache(StaticCache):
    """
    Custom KV Cache implementation for ExecutorTorch that inherits from Hugging Face's StaticCache
    but uses custom operations for cache updates similar to ExecutorTorch's CustomStaticCache.
    """

    def __init__(
        self,
        config,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ):
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            layer_device_map=layer_device_map,
        )

        # make sure layer_device_map is none
        assert layer_device_map is None
        assert device is None or device == "cpu", "Device must be None or 'cpu'"

        # Create a list of CustomKVCache instances, one per layer
        self.kv_cache = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer_cache = CustomKVCache(
                max_batch_size=self.max_batch_size,
                max_context_length=self.max_cache_len,
                n_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.kv_cache.append(layer_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`
        using ExecutorTorch's CustomKVCache.

        Args:
            key_states (`torch.Tensor`):
                The new key states to cache. Shape: [batch_size, n_heads, seq_len, head_dim]
            value_states (`torch.Tensor`):
                The new value states to cache. Shape: [batch_size, n_heads, seq_len, head_dim]
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache update.

        Returns:
            A tuple containing the updated key and value states.
        """
        assert cache_kwargs is not None

        # Get cache position from cache_kwargs (used by StaticCache)
        cache_position = cache_kwargs.get("cache_position")
        assert cache_position is not None
        assert isinstance(cache_position, torch.Tensor)

        # Get the CustomKVCache instance for this layer
        layer_cache = self.kv_cache[layer_idx]

        # Use the CustomKVCache's update method
        # CustomKVCache expects input_pos, k_val, v_val and handles the transpose internally
        k_out, v_out = layer_cache.update(
            input_pos=cache_position,
            k_val=key_states,
            v_val=value_states,
        )

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Occupied cache == any slot in the 2nd dim (sequence length) holds a non-zero value
        # This is different from StaticCache which checks the 3rd dim
        if layer_idx is None:
            layer_idx = 0
        return (self.kv_cache[layer_idx].k_cache[0, :, 0].any(dim=-1)).sum()

    @classmethod
    def from_legacy_cache(
        cls,
        config,
        legacy_cache,
        max_cache_len=None,
        device=None,
        dtype=None,
    ):
        """
        Create an ETCustomStaticCache from a legacy cache implementation.

        Args:
            config: The model configuration
            legacy_cache: The legacy cache implementation
            max_cache_len: The maximum cache length
            device: The device for the new cache
            dtype: The data type for the new cache

        Returns:
            A new ETCustomStaticCache instance
        """
        assert hasattr(legacy_cache, "k_cache") and hasattr(legacy_cache, "v_cache")
        # Extract dimensions from the legacy cache
        assert len(legacy_cache.k_cache.shape) == 4
        if legacy_cache.k_cache.shape[1] == legacy_cache.n_heads:
            # Shape is [batch_size, n_heads, seq_len, head_dim]
            max_batch_size = legacy_cache.k_cache.shape[0]
        else:
            # Shape is [batch_size, seq_len, n_heads, head_dim]
            max_batch_size = legacy_cache.k_cache.shape[0]

        # Use the legacy cache's device and dtype if not specified
        if device is None and hasattr(legacy_cache, "device"):
            device = legacy_cache.device
        elif device is None and hasattr(legacy_cache.k_cache, "device"):
            device = legacy_cache.k_cache.device

        if dtype is None and hasattr(legacy_cache, "dtype"):
            dtype = legacy_cache.dtype
        elif dtype is None and hasattr(legacy_cache.k_cache, "dtype"):
            dtype = legacy_cache.k_cache.dtype

        assert device is None or device == "cpu"
        assert dtype is None or dtype == torch.float32

        # Use the legacy cache's max_seq_len if max_cache_len is not specified
        if max_cache_len is None and hasattr(legacy_cache, "max_seq_len"):
            max_cache_len = legacy_cache.max_seq_len
        elif max_cache_len is None and hasattr(legacy_cache, "max_cache_len"):
            max_cache_len = legacy_cache.max_cache_len

        return cls(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )


# Need to figure out if I have to inherit from HybridCache or StaticCache
class ETCustomHybridCache(HybridCache):
    """
    Custom Hybrid KV Cache implementation for ExecutorTorch that inherits from Hugging Face's HybridCache
    but uses ExecutorTorch's CustomKVCache for global layers and CustomRingKVCache for sliding window layers.
    """

    def __init__(
        self,
        config,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ):
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            layer_device_map=layer_device_map,
        )

        # make sure layer_device_map is none
        assert layer_device_map is None
        assert device is None or device == "cpu", "Device must be None or 'cpu'"

        self.cache_position = None
        # Create a list of cache instances, one per layer
        # Use CustomKVCache for global layers and CustomRingKVCache for sliding window layers
        self.kv_cache = torch.nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            # newer version of transfomer has is_sliding defined
            # for HybridCache
            if self.is_sliding[layer_idx]:
                # This is a sliding window layer
                layer_cache = CustomRingKVCache(
                    max_batch_size=self.max_batch_size,
                    max_context_length=self.sliding_window_len,
                    n_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    dtype=dtype,
                )
            else:
                layer_cache = CustomKVCache(
                    max_batch_size=self.max_batch_size,
                    max_context_length=self.max_cache_len,
                    n_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    dtype=dtype,
                )
            self.kv_cache.append(layer_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`
        using ExecutorTorch's CustomKVCache or CustomRingKVCache depending on the layer type.

        Args:
            key_states (`torch.Tensor`):
                The new key states to cache. Shape: [batch_size, n_heads, seq_len, head_dim]
            value_states (`torch.Tensor`):
                The new value states to cache. Shape: [batch_size, n_heads, seq_len, head_dim]
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache update.

        Returns:
            A tuple containing the updated key and value states.
        """
        assert cache_kwargs is not None

        # Get cache position from cache_kwargs (used by HybridCache)
        cache_position = cache_kwargs.get("cache_position")
        assert cache_position is not None
        assert isinstance(cache_position, torch.Tensor)
        self.cache_position = cache_position

        # Get the cache instance for this layer (either CustomKVCache or CustomRingKVCache)
        layer_cache = self.kv_cache[layer_idx]

        # Use the cache's update method
        # Both CustomKVCache and CustomRingKVCache have the same update interface
        k_out, v_out = layer_cache.update(
            input_pos=cache_position,
            k_val=key_states,
            v_val=value_states,
        )

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is None:
            layer_idx = 0

        # For CustomRingKVCache, we need to handle the sequence length differently
        layer_cache = self.kv_cache[layer_idx]
        if self.is_sliding[layer_idx]:
            # CustomRingKVCache cache_position_manager which
            # maintains cache position for each slot in the kv cache
            # we return the max position + 1 to indicate max position
            # seen so far. Not sure if thats the correct interpretation
            # of sequence length
            return layer_cache.cache_positions_manager.cache_positions.max().item() + 1
        return (layer_cache.k_cache[0, :, 0].any(dim=-1)).sum()

    def get_layer_cache(self, layer_idx: int):
        """
        Get the cache for a specific layer. This method is dynamo-traceable.

        Args:
            layer_idx (int): The layer index

        Returns:
            The cache instance for the specified layer (CustomKVCache or CustomRingKVCache)
        """
        return self.kv_cache[layer_idx]


def replace_with_et_custom_kv_cache(module, config, generation_config, cache_dtype):
    """
    Replace all KV caches in the module with ETCustomStaticCache.
    This modifies the model in place.

    Args:
        module: The module to modify
        config: The model configuration

    Returns:
        The modified module
    """
    # Recursively replace KV caches
    return _replace_with_et_custom_kv_cache(
        module, config, generation_config, cache_dtype
    )


def _replace_with_et_custom_kv_cache(module, config, generation_config, cache_dtype):
    """
    Helper function to recursively replace KV caches in the module.

    Args:
        module: The module to modify
        config: The model configuration

    Returns:
        The modified module
    """
    # Check if module has static_cache (TorchExportableModuleWithStaticCache)
    if hasattr(module, "static_cache"):
        assert isinstance(
            module.static_cache, StaticCache
        ), f"Expected StaticCache, got {type(module.static_cache)}"

        # TODO: Add replace_cache to exported module
        # in transformer's executorch.py
        if getattr(module, "replace_cache", None) is not None:
            static_cache = ETCustomStaticCache(
                config=config,
                max_batch_size=generation_config.cache_config.batch_size,
                max_cache_len=generation_config.cache_config.max_cache_len,
                device=generation_config.cache_config.device,
                dtype=cache_dtype,
            )
            module.replace_cache(static_cache)
        else:
            module.static_cache = ETCustomStaticCache(
                config=config,
                max_batch_size=generation_config.cache_config.batch_size,
                max_cache_len=generation_config.cache_config.max_cache_len,
                device=generation_config.cache_config.device,
                dtype=cache_dtype,
            )
            # Dont know why we need to this even though
            # CustomKVCache registers the attributes
            for i in range(len(module.static_cache.kv_cache)):
                setattr(
                    module, f"key_cache_{i}", module.static_cache.kv_cache[i].k_cache
                )
                setattr(
                    module, f"value_cache_{i}", module.static_cache.kv_cache[i].v_cache
                )

    # Check if module has cache (TorchExportableModuleWithHybridCache)
    elif hasattr(module, "cache"):
        assert isinstance(
            module.cache, HybridCache
        ), f"Expected HybridCache, got {type(module.cache)}"

        # Replace with ETCustomHybridCache
        if getattr(module, "replace_cache", None) is not None:
            hybrid_cache = ETCustomHybridCache(
                config=config,
                max_batch_size=generation_config.cache_config.batch_size,
                max_cache_len=generation_config.cache_config.max_cache_len,
                device=generation_config.cache_config.device,
                dtype=cache_dtype,
            )
            module.replace_cache(hybrid_cache)
        else:
            module.cache = ETCustomHybridCache(
                config=config,
                max_batch_size=generation_config.cache_config.batch_size,
                max_cache_len=generation_config.cache_config.max_cache_len,
                device=generation_config.cache_config.device,
                dtype=cache_dtype,
            )
            # Register cache attributes for each layer
            for i in range(len(module.cache.kv_cache)):
                setattr(module, f"key_cache_{i}", module.cache.kv_cache[i].k_cache)
                setattr(module, f"value_cache_{i}", module.cache.kv_cache[i].v_cache)
                if module.cache.is_sliding[i]:
                    # Register cache_positions as buffer for sliding window layers
                    # This prevents it from being traced as a constant
                    module.register_buffer(
                        f"cache_positions_{i}",
                        module.cache.kv_cache[
                            i
                        ].cache_positions_manager.cache_positions,
                        persistent=False,
                    )
    else:
        raise ValueError(
            "Module must have either 'static_cache' (TorchExportableModuleWithStaticCache) "
            "or 'cache' (TorchExportableModuleWithHybridCache) attribute"
        )

    return module
