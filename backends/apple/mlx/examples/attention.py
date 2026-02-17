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
    from executorch.backends.apple.mlx.examples.attention import register_mlx_attention

    register_mlx_attention()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="mlx",
    )
"""

from typing import Callable, Optional, Tuple, Union

import executorch.backends.apple.mlx.custom_ops as _mlx_custom_ops  # noqa: F401

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
        torch._check_is_size(start_pos)
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
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Optional[Callable] = None,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    allow_torch_fix: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Returns None — the custom SDPA op handles causal masking internally.

    Returning None avoids materializing a mask tensor during export, which
    would create a bounded tensor that fails at runtime with longer sequences.
    """
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
