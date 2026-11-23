# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face attention over the off-graph KV cache, for any backend.

``register_offgraph_attention`` registers an HF attention implementation whose
layers each emit one ``kvcache::update_and_attend``. The model must run with
``use_cache=False`` so key/value are this step's projections; the op places
them by ``position_ids`` and masks causally itself, so the registered mask
function returns None.
"""

from typing import Callable, Optional, Tuple, Union

import executorch.extension.llm.cache.update_and_attend  # noqa: F401
import torch


def _cache_id(module: torch.nn.Module) -> int:
    """Which cache a layer addresses -- a cache id, not a model layer index.

    A KV-sharing layer (gemma 4's YOCO) computes no k/v of its own; HuggingFace
    hands it the k/v of the last non-sharing layer of the same attention type.
    Pointing it at that donor's cache makes its write repeat what the donor
    already stored and its read the shared history, so only donors own a cache.
    """
    if not getattr(module, "is_kv_shared_layer", False):
        return module.layer_idx
    cfg = module.config
    first_shared = cfg.num_hidden_layers - getattr(cfg, "num_kv_shared_layers", 0)
    donors = list(cfg.layer_types[:first_shared])
    # Must agree with how Gemma4TextAttention derives `store_full_length_kv`.
    return len(donors) - 1 - donors[::-1].index(cfg.layer_types[module.layer_idx])


def offgraph_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [B, num_heads, q_len, head_dim]
    key: torch.Tensor,  # [B, num_kv_heads, q_len, head_dim], this step only
    value: torch.Tensor,  # [B, num_kv_heads, q_len, head_dim], this step only
    attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa: F821
    position_ids: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Returns (output, None) with output [B, q_len, num_heads, head_dim]."""
    # Reject only what changes the result; HF forwards open-ended kwargs.
    if kwargs.get("dropout"):
        raise ValueError("off-graph attention does not support dropout")
    if softcap is not None:
        raise ValueError("off-graph attention does not support softcap")
    if head_mask is not None:
        raise ValueError("off-graph attention does not support head_mask")
    assert (
        position_ids is not None
    ), "position_ids must be provided to place tokens in the off-graph cache"
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
    """Returns None: the attention op masks itself, so no bounded mask tensor is traced."""
    return None


def register_offgraph_attention(name: str = "offgraph") -> None:
    """Register off-graph attention as HF ``attn_implementation=name``."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(name, offgraph_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, sdpa_mask_passthrough)
