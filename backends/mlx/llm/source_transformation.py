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
from typing import Callable, Sequence

import torch
import torch.nn as nn

from executorch.backends.mlx.llm.cache import (
    HFStaticCache,
    KVCache,
    KVCacheLayerConfig,
    resolve_hf_cache_layout,
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


def replace_hf_cache_with_mlx_in_graph_cache(
    module: nn.Module,
    layer_configs: Sequence[KVCacheLayerConfig],
    *,
    max_cache_len: int,
    max_write_len: int,
    max_batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Install full/ring caches from resolved cache-owner geometry, in place.

    The wrapper's ``static_cache`` or ``cache`` must be None or a StaticCache.
    All geometry is validated before allocation or wrapper mutation. Cache
    tensors are registered as nonpersistent buffers for export capture.
    """
    from transformers.cache_utils import StaticCache

    if hasattr(module, "static_cache"):
        attr_name = "static_cache"
    elif hasattr(module, "cache"):
        attr_name = "cache"
    else:
        raise ValueError("Module must have 'static_cache' or 'cache' attribute")
    original_cache = getattr(module, attr_name)
    if original_cache is not None and not isinstance(original_cache, StaticCache):
        raise ValueError(
            f"module.{attr_name} must be None or a StaticCache, "
            f"got {type(original_cache)}"
        )

    mlx_cache = HFStaticCache.from_layer_configs(
        layer_configs,
        max_cache_len=max_cache_len,
        max_write_len=max_write_len,
        max_batch_size=max_batch_size,
        dtype=dtype,
    )
    for i, layer in enumerate(mlx_cache.layers):
        for name, tensor in (
            (f"key_cache_{i}", layer.keys),
            (f"value_cache_{i}", layer.values),
            (f"cumulative_length_{i}", layer.cumulative_length),
        ):
            # Older wrappers may expose plain tensor attributes, not buffers.
            if hasattr(module, name) and name not in module._buffers:
                delattr(module, name)
            module.register_buffer(name, tensor, persistent=False)
    setattr(module, attr_name, mlx_cache)
    return module


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
    _, num_heads, head_dims = resolve_hf_cache_layout(config)
    context_length = (
        max_cache_len
        if max_cache_len is not None
        else getattr(config.get_text_config(), "max_position_embeddings", 2048)
    )
    return replace_hf_cache_with_mlx_in_graph_cache(
        module,
        [
            KVCacheLayerConfig(num_kv_heads=num_head, head_dim=head_dim, window_size=0)
            for num_head, head_dim in zip(num_heads, head_dims)
        ],
        max_batch_size=max_batch_size,
        max_cache_len=context_length,
        max_write_len=context_length,
        dtype=dtype,
    )


def replace_hf_cache_with_mlx_ring_buffer(
    module: nn.Module,
    config,
    max_batch_size: int = 1,
    window_size: int = 512,
    max_cache_len: int | None = None,
    dtype: torch.dtype = torch.float32,
    max_write_len: int | None = None,
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
        window_size: Sliding window size (capacity of the sliding-layer rings)
        max_cache_len: Capacity of the full-attention layers; defaults to
            ``window_size``, which is only correct for models with no
            full-attention layers
        dtype: Cache tensor dtype
        max_write_len: Largest single write the sliding layers must accept;
            sizes each ring as ``window_size + max_write_len - 1``. Defaults to
            ``window_size``.

    Raises:
        ValueError: If module has no recognized cache attribute, or if
            ``max_write_len`` exceeds ``window_size``
    """
    layer_types, num_heads, head_dims = resolve_hf_cache_layout(config)
    full_cache_len = max_cache_len if max_cache_len is not None else window_size
    ring_max_write = max_write_len if max_write_len is not None else window_size
    return replace_hf_cache_with_mlx_in_graph_cache(
        module,
        [
            KVCacheLayerConfig(
                num_kv_heads=num_head,
                head_dim=head_dim,
                window_size=window_size if layer_type == "sliding_attention" else 0,
            )
            for layer_type, num_head, head_dim in zip(layer_types, num_heads, head_dims)
        ],
        max_batch_size=max_batch_size,
        max_cache_len=full_cache_len,
        max_write_len=ring_max_write,
        dtype=dtype,
    )


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
