#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MLX-optimized attention for HuggingFace models.

Registers a custom attention implementation ("mlx") with HuggingFace's
attention interface, following the same pattern as optimum-executorch's
custom_sdpa:

1. Mask function returns None (custom op handles causal masking internally)
2. Attention function extracts start_pos from position_ids[0][0]
3. mlx::custom_sdpa receives full K/V cache + start_pos, slices K/V internally
4. MLX pattern handler serializes custom_sdpa as SliceNode(K), SliceNode(V), SdpaNode

Usage:
    from executorch.backends.mlx.llm.hf_attention import register_mlx_attention

    register_mlx_attention()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="mlx",
    )
"""

from typing import Callable, Optional, Tuple, Union

import executorch.backends.mlx.custom_ops as _mlx_custom_ops  # noqa: F401

import torch


def mlx_sdpa_with_start_pos_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [B, num_heads, seq_len, head_dim] - BHSD
    key: torch.Tensor,  # [B, num_kv_heads, kv_len, head_dim] - BHSD (full cache)
    value: torch.Tensor,  # [B, num_kv_heads, kv_len, head_dim] - BHSD (full cache)
    attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa: F821
    position_ids: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    MLX-optimized SDPA following optimum-executorch's custom_sdpa pattern.

    Extracts start_pos from position_ids, then delegates to mlx::custom_sdpa
    which handles K/V cache slicing, GQA expansion, and causal masking.

    Returns (output, None) where output is [B, seq_len, num_heads, head_dim] (BSHD).
    """
    kwargs.pop("is_causal", None)
    is_causal = getattr(module, "is_causal", True)

    if is_causal:
        assert (
            position_ids is not None
        ), "position_ids must be provided to find start position for causal attention"
        start_pos = position_ids[0][0].item()
        seq_len = query.shape[2]
        torch._check(start_pos >= 0)
        torch._check(start_pos + seq_len <= key.shape[2])
        attn_mask = None
    else:
        start_pos = 0
        attn_mask = attention_mask

    output = torch.ops.mlx.custom_sdpa(
        query,
        key,
        value,
        start_pos=start_pos,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scaling,
    )

    # Transpose BHSD → BSHD for HF
    return output.transpose(1, 2).contiguous(), None


def sdpa_mask_passthrough(
    batch_size: int,
    cache_position: Optional[torch.Tensor] = None,
    q_length: Optional[int] = None,
    kv_length: Optional[int] = None,
    q_offset: Optional[Union[int, torch.Tensor]] = None,
    kv_offset: int = 0,
    mask_function: Optional[Callable] = None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_torch_fix: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    """Returns None — custom SDPA handles causal masking, avoiding bounded mask tensors."""
    return None


def register_mlx_attention(name: str = "mlx") -> None:
    """
    Register MLX attention with HuggingFace's attention interfaces.

    After registration, models can use MLX attention via:
        model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="mlx")
    """
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register(name, mlx_sdpa_with_start_pos_forward)
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, sdpa_mask_passthrough)

    except ImportError:
        raise ImportError(
            "transformers is not installed. Please install it: pip install transformers"
        )


def get_mlx_sliding_window_sdpa(exportable_module) -> Callable:
    """
    Create a closure-based SDPA function for sliding window attention.

    Following optimum-executorch's pattern, the returned function captures
    the model reference so it can access ring buffer caches at runtime to
    create attention masks lazily — avoiding torch.export tracing issues.

    Args:
        exportable_module: The model module containing .cache (HFStaticCache
            or similar) with ring buffer layers accessible via .kv_cache[layer_idx].

    Returns:
        Attention function compatible with HuggingFace's attention interface.
    """

    def _resolve_cache_layer_idx(module: torch.nn.Module, cache) -> Optional[int]:
        """
        Map a transformer layer index to the backing cache slot index.

        Hybrid/shared-KV models like Gemma 4 only allocate cache entries for the
        non-shared KV layers. Shared layers expose `kv_shared_layer_index`, which
        points at the earlier cache-producing layer they reuse.
        """
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return None

        if layer_idx < len(cache.kv_cache):
            return layer_idx

        shared_layer_idx = getattr(module, "kv_shared_layer_index", None)
        if shared_layer_idx is not None and shared_layer_idx < len(cache.kv_cache):
            return shared_layer_idx

        return None

    def _sliding_window_sdpa_forward(
        module: torch.nn.Module,
        query: torch.Tensor,  # [B, num_heads, seq_len, head_dim] - BHSD
        key: torch.Tensor,  # [B, num_kv_heads, window_size, head_dim] - BHSD
        value: torch.Tensor,  # [B, num_kv_heads, window_size, head_dim] - BHSD
        attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa: F821
        position_ids: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        MLX sliding window SDPA using ring buffer KV cache.

        Creates the attention mask lazily by reaching into the ring buffer
        cache via the captured model reference. This keeps mask creation
        in Python (not in the traced graph).

        Uses is_causal=False since the mask handles both causality and windowing.
        """
        from executorch.backends.mlx.llm.cache import RingBufferKVCache

        layer_idx = getattr(module, "layer_idx", None)
        seq_len = query.shape[2]
        attn_mask = None
        start_pos = 0

        layer_cache = None
        if layer_idx is not None and position_ids is not None:
            start_pos = position_ids[0][0].item()

            # Reach into the model's cache to find the ring buffer for this layer.
            # TorchExportableModuleWithHybridCache stores .cache (standard path).
            cache = getattr(exportable_module, "cache", None)

            if cache is not None:
                cache_layer_idx = _resolve_cache_layer_idx(module, cache)
                if cache_layer_idx is not None:
                    layer_cache = cache.kv_cache[cache_layer_idx]
                if isinstance(layer_cache, RingBufferKVCache):
                    attn_mask = layer_cache.create_sliding_window_mask(
                        start_pos, seq_len
                    )
                    # Override start_pos so custom_sdpa slices the full buffer:
                    # stop_pos = start_pos + seq_len = buffer_size
                    start_pos = layer_cache.buffer_size - seq_len

        # Hybrid models use one global HF attention implementation. Sliding
        # layers need the ring-buffer mask path, while full-attention layers
        # should keep the regular causal SDPA path even under the same hook.
        if attn_mask is None:
            return mlx_sdpa_with_start_pos_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                position_ids=position_ids,
                scaling=scaling,
                **kwargs,
            )

        output = torch.ops.mlx.custom_sdpa(
            query,
            key,
            value,
            start_pos=start_pos,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scaling,
        )

        # Transpose BHSD → BSHD for HF
        return output.transpose(1, 2).contiguous(), None

    return _sliding_window_sdpa_forward


def register_mlx_sliding_window_attention(
    exportable_module, name: str = "mlx_sliding_window"
) -> None:
    """Register MLX sliding window attention with HuggingFace's attention interfaces."""
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        sdpa_fn = get_mlx_sliding_window_sdpa(exportable_module)
        ALL_ATTENTION_FUNCTIONS.register(name, sdpa_fn)
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, sdpa_mask_passthrough)

    except ImportError:
        raise ImportError(
            "transformers is not installed. Please install it: pip install transformers"
        )
