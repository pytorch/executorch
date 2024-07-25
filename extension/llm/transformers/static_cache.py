# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PretrainedConfig, StaticCache


class ETStaticCache(torch.nn.Module, StaticCache):
    """
    Static Cache class to be used with `torch.compile(model)`.
    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: int,
        device,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = (
            config.max_position_embeddings if max_cache_len is None else max_cache_len
        )
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (
            max_batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )
        for idx in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            self.register_buffer(
                f"key_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=device)
            )
            self.register_buffer(
                f"val_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=device)
            )
            key_cache = getattr(self, f"key_cache_{idx}")
            val_cache = getattr(self, f"val_cache_{idx}")
            torch._dynamo.mark_static_address(key_cache)
            torch._dynamo.mark_static_address(val_cache)
            self.key_cache.append(key_cache)
            self.value_cache.append(val_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.
        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states
        seq_len = self.get_seq_length(layer_idx)
        return (
            k_out[:, :, torch.arange(0, seq_len, device=k_out.device), :],
            v_out[:, :, torch.arange(0, seq_len, device=v_out.device), :],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum().item()

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    def from_legacy_cache(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                self.update(key_states, value_states, layer_idx, cache_kwargs)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)
