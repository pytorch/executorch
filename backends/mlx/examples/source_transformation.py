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
versions using mlx::kv_cache_update and mlx::custom_sdpa custom ops.

Transforms:
- replace_et_kv_cache_with_mlx: ET KVCache → ETKVCache (for ET Llama models)
- replace_hf_cache_with_mlx: HF StaticCache → HFStaticCache (for HF models)
- replace_rope_with_mlx: ET Rope → MLXRope
- transform_attention_mha_to_mlx: ET AttentionMHA → MLXAttentionMHA

Usage:
    from executorch.backends.mlx.examples.source_transformation import (
        get_mlx_source_transforms,
    )

    transforms = get_mlx_source_transforms()
    for transform in transforms:
        model = transform(model)
"""

import logging
from typing import Callable, List

import torch
import torch.nn as nn

from executorch.backends.mlx.examples.cache import ETKVCache, HFStaticCache

logger = logging.getLogger(__name__)


def replace_et_kv_cache_with_mlx(
    module: nn.Module, dtype: torch.dtype = None
) -> nn.Module:
    """
    Replace ET's KVCache with MLX-optimized ETKVCache.

    Recursively finds all KVCache instances (from examples/models/llama/attention.py)
    and replaces them with ETKVCache, which uses mlx::kv_cache_update instead of
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

    def _replace_recursive(parent: nn.Module) -> int:
        replaced = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, ETKVCache_Original):
                cache_dtype = dtype if dtype is not None else child.k_cache.dtype
                mlx_cache = ETKVCache(
                    max_batch_size=child.max_batch_size,
                    max_context_length=child.max_context_length,
                    n_heads=child.n_heads,
                    head_dim=child.head_dim,
                    enable_dynamic_shape=child.enable_dynamic_shape,
                    dtype=cache_dtype,
                )
                setattr(parent, name, mlx_cache)
                replaced += 1
            else:
                replaced += _replace_recursive(child)
        return replaced

    count = _replace_recursive(module)
    if count > 0:
        logger.info(f"Replaced {count} KVCache → ETKVCache (dtype={dtype})")

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
    from transformers.cache_utils import StaticCache

    mlx_cache = HFStaticCache(
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=max_cache_len,
        dtype=dtype,
    )

    def _install_cache(attr_name):
        setattr(module, attr_name, mlx_cache)
        for i, layer_cache in enumerate(mlx_cache.kv_cache):
            setattr(module, f"key_cache_{i}", layer_cache.k_cache)
            setattr(module, f"value_cache_{i}", layer_cache.v_cache)

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
                torch._check_is_size(input_pos_item)
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


def replace_rope_with_mlx(module: nn.Module) -> nn.Module:
    """Replace ET's Rope with MLXRope throughout the model."""
    from executorch.examples.models.llama.rope import Rope

    def _replace_recursive(parent: nn.Module) -> int:
        replaced = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, Rope):
                setattr(parent, name, MLXRope(child))
                replaced += 1
            else:
                replaced += _replace_recursive(child)
        return replaced

    count = _replace_recursive(module)
    logger.info(f"Replaced {count} Rope → MLXRope")
    return module


def transform_attention_mha_to_mlx(
    module: nn.Module, dtype: torch.dtype = None
) -> nn.Module:
    """
    Replace AttentionMHA with MLXAttentionMHA throughout the model.

    Shares weight references (wq, wk, wv, wo, rope, norm) from the original
    and creates a fresh ETKVCache for each attention layer.

    Args:
        module: Model to modify (in place)
        dtype: Optional dtype for KV cache. If None, inferred from original.
    """
    from executorch.backends.mlx.examples.et_attention import MLXAttentionMHA
    from executorch.examples.models.llama.attention import AttentionMHA

    def _replace_recursive(parent: nn.Module) -> int:
        replaced = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, AttentionMHA):
                setattr(
                    parent, name, MLXAttentionMHA.from_attention_mha(child, dtype=dtype)
                )
                replaced += 1
            else:
                replaced += _replace_recursive(child)
        return replaced

    count = _replace_recursive(module)
    if count > 0:
        logger.info(
            f"Replaced {count} AttentionMHA → MLXAttentionMHA (cache dtype={dtype})"
        )
    else:
        logger.warning("No AttentionMHA instances found")

    return module


def preextract_start_pos(module: nn.Module) -> nn.Module:
    """
    Wrap Transformer.forward to extract start_pos = input_pos[0].item() once
    and inject it into attn_options, so per-layer attention doesn't repeat it.
    """
    from executorch.examples.models.llama.llama_transformer import Transformer

    if not isinstance(module, Transformer):
        logger.warning("preextract_start_pos: module is not a Transformer, skipping")
        return module

    original_forward = module.forward

    def wrapped_forward(
        tokens=None,
        attn_options=None,
        h=None,
    ):
        if attn_options is not None and "input_pos" in attn_options:
            input_pos = attn_options["input_pos"]
            attn_options = attn_options.copy()
            attn_options["start_pos"] = input_pos[0].item()
        return original_forward(tokens=tokens, attn_options=attn_options, h=h)

    module.forward = wrapped_forward
    logger.info("Wrapped Transformer.forward to pre-extract start_pos")
    return module


class FunctionalRMSNorm(nn.Module):
    """RMSNorm using F.rms_norm for efficient MLX execution.

    Replaces manual variance + rsqrt computation with the aten rms_norm op,
    which the MLX backend maps directly to fast::rms_norm.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.weight = original.weight
        self.eps = float(original.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x, (self.weight.shape[0],), self.weight, self.eps
        )


def replace_rms_norm_with_functional(module: nn.Module) -> nn.Module:
    """Replace RMSNorm with FunctionalRMSNorm using F.rms_norm."""
    from executorch.examples.models.llama.norm import RMSNorm

    def _replace_recursive(parent: nn.Module) -> int:
        replaced = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, RMSNorm):
                setattr(parent, name, FunctionalRMSNorm(child))
                replaced += 1
            else:
                replaced += _replace_recursive(child)
        return replaced

    count = _replace_recursive(module)
    if count > 0:
        logger.info(f"Replaced {count} RMSNorm → FunctionalRMSNorm (F.rms_norm)")
    return module


def get_mlx_source_transforms(
    use_mlx_kv_cache: bool = True,
    use_mlx_rope: bool = False,
    use_mlx_attention: bool = True,
    kv_cache_dtype: torch.dtype = None,
) -> List[Callable[[nn.Module], nn.Module]]:
    """
    Get the list of source transforms for MLX export.

    Args:
        use_mlx_kv_cache: Replace ET KVCache with ETKVCache (default True).
            When use_mlx_attention is True, attention replacement already
            includes ETKVCache; this catches any remaining KVCache instances.
        use_mlx_rope: Replace ET Rope with MLXRope (default False)
        use_mlx_attention: Replace AttentionMHA with MLXAttentionMHA (default True)
        kv_cache_dtype: Optional dtype for KV cache tensors

    Returns:
        List of transform functions to apply sequentially
    """
    from functools import partial

    transforms = []

    # Attention replacement first — replaces entire AttentionMHA including KV cache
    if use_mlx_attention:
        transforms.append(partial(transform_attention_mha_to_mlx, dtype=kv_cache_dtype))

    # Standalone KV cache replacement catches remaining instances
    if use_mlx_kv_cache:
        transforms.append(partial(replace_et_kv_cache_with_mlx, dtype=kv_cache_dtype))

    if use_mlx_rope:
        transforms.append(replace_rope_with_mlx)

    # Pre-extract start_pos once in Transformer.forward rather than per-layer
    if use_mlx_attention:
        transforms.append(preextract_start_pos)

    # Replace RMSNorm with F.rms_norm for fused execution
    transforms.append(replace_rms_norm_with_functional)

    return transforms
