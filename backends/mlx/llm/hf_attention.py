#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MLX-optimized attention for HuggingFace models.

Two implementations are registered with HuggingFace's attention interface,
differing in where the KV cache lives. Both register a mask function returning
None, since the op masks internally and a mask tensor would be traced at a
fixed size.

"mlx" (register_mlx_attention) keeps the cache in the graph: the model runs
with a StaticCache, so key/value are the full history, and the attention
function extracts start_pos from position_ids and hands both to
mlx::custom_sdpa, which slices K/V and applies causal masking. The MLX pattern
handler serializes it as SliceNode(K), SliceNode(V), SdpaNode.

"mlx_offgraph" (register_mlx_offgraph_attention) keeps the cache out of the
graph: the model must run with use_cache=False, so key/value are only this
step's projections, and each layer emits one kvcache::update_and_attend fed the
token positions. The cache is created and bound at run time via a cache_key.
OffGraphExportWrapper supplies the (input_ids, cache_position) signature that
path needs.

Usage:
    from executorch.backends.mlx.llm.hf_attention import register_mlx_attention

    register_mlx_attention()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="mlx",   # or "mlx_offgraph"
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
    # Refuse only what would silently change the result. Anything else in
    # kwargs is ignored: HuggingFace forwards model-level arguments down into
    # attention -- gemma 4 sends `labels` -- so the set is open-ended and
    # rejecting the unknown breaks models that pass something harmless.
    if kwargs.get("dropout"):
        raise ValueError("mlx attention does not support dropout")
    if softcap is not None:
        raise ValueError("mlx attention does not support softcap")
    if head_mask is not None:
        raise ValueError("mlx attention does not support head_mask")
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


def _cache_id(module: torch.nn.Module) -> int:
    """Which cache a layer addresses -- a cache id, not a model layer index.

    A KV-sharing layer (gemma 4's YOCO) computes no k/v of its own; HuggingFace
    hands it the k/v of the last non-sharing layer of the same attention type.
    Pointing it at that donor's cache makes its write repeat what the donor
    already stored and its read the shared history, so only donors own a cache
    (15 rather than 35 for gemma-4-E2B).
    """
    if not getattr(module, "is_kv_shared_layer", False):
        return module.layer_idx
    cfg = module.config
    first_shared = cfg.num_hidden_layers - getattr(cfg, "num_kv_shared_layers", 0)
    donors = list(cfg.layer_types[:first_shared])
    # Last index in `donors` of this layer's attention type. Kept identical to
    # how Gemma4TextAttention derives `store_full_length_kv`, since that is the
    # layer whose k/v we are handed; the two must agree.
    return len(donors) - 1 - donors[::-1].index(cfg.layer_types[module.layer_idx])


def mlx_offgraph_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [B, num_heads, q_len, head_dim] - BHSD
    key: torch.Tensor,  # [B, num_kv_heads, q_len, head_dim] - BHSD (this step)
    value: torch.Tensor,  # [B, num_kv_heads, q_len, head_dim] - BHSD (this step)
    attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa: F821
    position_ids: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Attention over the off-graph KV cache (kvcache::update_and_attend).

    Unlike the mlx::custom_sdpa path, no cache is in the graph: the model must be
    run with use_cache=False so key/value are this step's projections, not full
    history. The op owns the cache, placing/reading by `position` and masking
    itself, so this is causal-only and never materializes a mask tensor.

    Returns (output, None) where output is [B, q_len, num_heads, head_dim] (BSHD).
    """
    # As above: reject only what changes the result, ignore the rest.
    if kwargs.get("dropout"):
        raise ValueError("mlx_offgraph attention does not support dropout")
    if softcap is not None:
        raise ValueError("mlx_offgraph attention does not support softcap")
    if head_mask is not None:
        raise ValueError("mlx_offgraph attention does not support head_mask")
    assert (
        position_ids is not None
    ), "position_ids must be provided to place tokens in the off-graph cache"
    # Per-token absolute positions [q_len, 1]; the op places + masks from these.
    position = position_ids[0].reshape(-1, 1)
    assert scaling is not None, "scaling must be provided by the attention module"

    output = torch.ops.kvcache.update_and_attend(
        query,
        key,
        value,
        position,
        layer_id=_cache_id(module),
        scale=scaling,
        out_dtype=query.dtype,
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


class OffGraphExportWrapper(torch.nn.Module):
    """forward(input_ids, cache_position) -> logits, with no in-graph cache.

    The analog of TorchExportableModuleWithStaticCache for the off-graph op:
    runs the model with use_cache=False so each attention layer sees only this
    step's k/v (the op owns history), and exposes the (input_ids, cache_position)
    signature the runner drives.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        # Single sequence: the op takes [q_len, n_dims] positions and the
        # attention function reads position_ids[0], so a batch would be placed
        # at row 0's positions.
        assert input_ids.shape[0] == 1, "off-graph export supports batch size 1"
        return self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            # Pass positions rather than letting the model infer them: it derives
            # them from past_key_values.get_seq_length(), which is 0 here because
            # the cache is out of the graph, so every decode step would place its
            # token at position 0.
            position_ids=cache_position.unsqueeze(0),
            use_cache=False,
            past_key_values=None,
        ).logits


def register_mlx_offgraph_attention(name: str = "mlx_offgraph") -> None:
    """
    Register off-graph KV-cache attention with HuggingFace's attention interface.

    Models using attn_implementation="mlx_offgraph" must be exported with
    use_cache=False (no StaticCache): the cache lives outside the graph, bound at
    runtime via cache_key. Importing the op module registers the custom op.
    """
    from executorch.extension.llm.cache import update_and_attend  # noqa: F401

    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register(name, mlx_offgraph_attention_forward)
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
