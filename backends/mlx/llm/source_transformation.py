#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Source transformations for MLX backend export.

Provides transforms that replace standard model components with MLX-optimized
versions
"""

import logging
from typing import Callable

import torch
import torch.nn as nn

from executorch.backends.mlx.llm.cache import (
    HFStaticCache,
    KVCache,
    resolve_hf_cache_layout,
    RingBufferKVCache,
)

logger = logging.getLogger(__name__)


def _replace_modules(
    module: nn.Module,
    target_type: type,
    factory: Callable[[nn.Module], nn.Module],
    label: str,
) -> nn.Module:
    """Recursively replace all instances of target_type using factory."""

    def _recurse(parent: nn.Module) -> int:
        count = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, target_type):
                setattr(parent, name, factory(child))
                count += 1
            else:
                count += _recurse(child)
        return count

    count = _recurse(module)
    if count > 0:
        logger.info(f"Replaced {count} {label}")
    return module


def replace_et_kv_cache_with_mlx(
    module: nn.Module, dtype: torch.dtype = None
) -> nn.Module:
    """
    Replace ET's KVCache with MLX-optimized KVCache.

    Recursively finds all KVCache instances (from examples/models/llama/attention.py)
    and replaces them with KVCache, which uses mlx::kv_cache_update instead of
    unsupported index_put operations.

    Args:
        module: Model to modify (in place)
        dtype: Optional dtype for cache tensors. If None, uses original cache dtype.
    """
    try:
        from executorch.examples.models.llama.attention import (
            KVCache as ETKVCache_Original,
        )
    except ImportError:
        return module

    def _make_mlx_cache(child):
        cache_dtype = dtype if dtype is not None else child.k_cache.dtype
        return KVCache(
            max_batch_size=child.max_batch_size,
            max_context_length=child.max_context_length,
            n_heads=child.n_heads,
            head_dim=child.head_dim,
            enable_dynamic_shape=child.enable_dynamic_shape,
            dtype=cache_dtype,
        )

    return _replace_modules(
        module,
        ETKVCache_Original,
        _make_mlx_cache,
        f"KVCache → KVCache (dtype={dtype})",
    )


def replace_hf_cache_with_mlx(
    module: nn.Module,
    config,
    max_batch_size: int = 1,
    max_cache_len: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Replace HuggingFace's StaticCache with MLX-optimized HFStaticCache.

    Should be called on TorchExportableModuleWithStaticCache (from
    transformers.integrations.executorch), NOT on CausalLMExportableModule
    (from optimum-executorch).

    Args:
        module: HF exportable module with static_cache or cache attribute
        config: HF model config
        max_batch_size: Maximum batch size (default: 1)
        max_cache_len: Maximum cache length. If None, uses config.max_position_embeddings
        dtype: Cache tensor dtype (default: torch.float32)

    Raises:
        ValueError: If module has no recognized cache attribute
    """
    from transformers.cache_utils import StaticCache

    mlx_cache = HFStaticCache(
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=max_cache_len,
        dtype=dtype,
    )

    def _install_cache(attr_name):
        setattr(module, attr_name, mlx_cache)
        for i, (cache_layer, layer_cache) in enumerate(
            zip(mlx_cache.layers, mlx_cache.kv_cache)
        ):
            setattr(module, f"key_cache_{i}", layer_cache.k_cache)
            setattr(module, f"value_cache_{i}", layer_cache.v_cache)
            if hasattr(cache_layer, "cumulative_length"):
                setattr(
                    module,
                    f"cumulative_length_{i}",
                    cache_layer.cumulative_length,
                )

    if hasattr(module, "static_cache"):
        assert isinstance(
            module.static_cache, StaticCache
        ), f"Expected StaticCache, got {type(module.static_cache)}"
        _install_cache("static_cache")
    elif hasattr(module, "cache"):
        if isinstance(module.cache, StaticCache):
            _install_cache("cache")
        else:
            raise ValueError(
                f"module.cache is not a StaticCache, got {type(module.cache)}"
            )
    else:
        raise ValueError("Module must have 'static_cache' or 'cache' attribute")

    return module


def replace_hf_cache_with_mlx_ring_buffer(
    module: nn.Module,
    config,
    max_batch_size: int = 1,
    window_size: int = 512,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Replace HuggingFace's StaticCache with RingBufferKVCache for sliding window models.

    Creates a HFStaticCache-like structure where each layer uses a RingBufferKVCache
    instead of a linear KVCache. This enables infinite-length generation for models
    with sliding window attention (e.g., gemma).

    Args:
        module: HF exportable module with static_cache or cache attribute
        config: HF model config
        max_batch_size: Maximum batch size (default: 1)
        window_size: Sliding window size (cache capacity per layer)
        dtype: Cache tensor dtype

    Raises:
        ValueError: If module has no recognized cache attribute
    """
    from transformers.cache_utils import StaticCache

    # Create HFStaticCache with ring buffer layers
    mlx_cache = HFStaticCache(
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=window_size,
        dtype=dtype,
    )

    # Replace only the sliding-window cache entries with ring buffers, while
    # preserving full-attention entries as linear caches. Hybrid models like
    # Gemma 4 mix both layouts and can also vary head_dim per cache layer.
    layer_types, num_heads, head_dims = resolve_hf_cache_layout(config)
    num_cache_layers = len(mlx_cache.layers)
    num_ring_layers = 0
    for i, (layer_type, layer_num_heads, layer_head_dim) in enumerate(
        zip(layer_types, num_heads, head_dims)
    ):
        if layer_type != "sliding_attention":
            continue
        mlx_cache.kv_cache[i] = RingBufferKVCache(
            max_batch_size=max_batch_size,
            max_context_length=window_size,
            n_heads=layer_num_heads,
            head_dim=layer_head_dim,
            dtype=dtype,
        )
        num_ring_layers += 1

    def _install_cache(attr_name):
        setattr(module, attr_name, mlx_cache)
        for i, (cache_layer, layer_cache) in enumerate(
            zip(mlx_cache.layers, mlx_cache.kv_cache)
        ):
            setattr(module, f"key_cache_{i}", layer_cache.k_cache)
            setattr(module, f"value_cache_{i}", layer_cache.v_cache)
            if hasattr(cache_layer, "cumulative_length"):
                setattr(
                    module,
                    f"cumulative_length_{i}",
                    cache_layer.cumulative_length,
                )

    if hasattr(module, "static_cache"):
        assert isinstance(
            module.static_cache, StaticCache
        ), f"Expected StaticCache, got {type(module.static_cache)}"
        _install_cache("static_cache")
    elif hasattr(module, "cache"):
        if isinstance(module.cache, StaticCache):
            _install_cache("cache")
        else:
            raise ValueError(
                f"module.cache is not a StaticCache, got {type(module.cache)}"
            )
    else:
        raise ValueError("Module must have 'static_cache' or 'cache' attribute")

    logger.info(
        f"Installed hybrid MLX cache: {num_ring_layers} ring-buffer layers / "
        f"{num_cache_layers} total cache layers, window_size={window_size}"
    )

    return module


class MLXRope(nn.Module):
    """
    MLX-optimized Rotary Position Embedding.

    Wraps ET's Rope, currently delegating to the original implementation.
    Can be extended to use torch.ops.mlx.rope.
    """

    def __init__(self, original_rope: nn.Module):
        super().__init__()
        self.params = original_rope.params
        self.precompute_freqs_cis = original_rope.precompute_freqs_cis
        self.apply_rotary_emb = original_rope.apply_rotary_emb
        self.register_buffer("freqs_cos", original_rope.freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", original_rope.freqs_sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        return self.apply_rotary_emb(q, k, freqs_cos, freqs_sin)

    def get_freqs(self, input_pos, seq_len: int):
        if self.params.use_kv_cache:
            assert input_pos is not None
            if self.params.enable_dynamic_shape:
                input_pos_item = input_pos[-1].item()
                torch._check(input_pos_item >= 0)
                torch._check(input_pos_item < self.params.max_context_len)
                freqs_cos = self.freqs_cos.narrow(0, input_pos_item, seq_len)
                freqs_sin = self.freqs_sin.narrow(0, input_pos_item, seq_len)
            else:
                freqs_cos = self.freqs_cos[input_pos]
                freqs_sin = self.freqs_sin[input_pos]
        else:
            assert input_pos is None
            freqs_cos = self.freqs_cos[:seq_len]
            freqs_sin = self.freqs_sin[:seq_len]
        return freqs_cos, freqs_sin


def transform_attention_mha_to_mlx(
    module: nn.Module, dtype: torch.dtype = None
) -> nn.Module:
    """
    Replace AttentionMHA with MLXAttentionMHA throughout the model.

    Shares weight references (wq, wk, wv, wo, rope, norm) from the original
    and creates a fresh KVCache for each attention layer.

    Args:
        module: Model to modify (in place)
        dtype: Optional dtype for KV cache. If None, inferred from original.
    """
    from executorch.backends.mlx.llm.et_attention import MLXAttentionMHA
    from executorch.examples.models.llama.attention import AttentionMHA

    _replace_modules(
        module,
        AttentionMHA,
        lambda child: MLXAttentionMHA.from_attention_mha(child, dtype=dtype),
        f"AttentionMHA → MLXAttentionMHA (cache dtype={dtype})",
    )
    return module
