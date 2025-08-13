# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Optional, Tuple, Union

import torch
from executorch.extension.llm.custom_ops.custom_ops import custom_sdpa  # noqa


def custom_sdpa_with_start_pos_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Convert the hell out of the inputs to fp32 and back
    input_dtype = query.dtype
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    # Ignore the causal flag from kwargs but use the one in module
    kwargs.pop("is_causal", None)
    assert module.is_causal, "Current variant supports only causal attention"

    is_causal = module.is_causal
    if kwargs.get("is_sliding", False):
        is_causal = False
        attn_mask = attention_mask
        # start_pos is not important when using mask
        # instead of doing causal attention
        start_pos = 0
    else:
        attn_mask = None
        # Calculate the input pos from attention mask.
        # Branch out for float vs bool mask
        # assert attention_mask.dim() == 2, f"attention_mask must be a 2D matrix."
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        first_row_mask = attention_mask[0, :]
        # [0, 0, 0, 0, -inf, -inf, -inf, -inf], start_pos = 3
        start_pos = torch.argmin(first_row_mask.to(torch.long)).item() - 1

    output = torch.ops.llama.custom_sdpa(
        query,
        key,
        value,
        start_pos=start_pos,
        attn_mask=attn_mask,
        drpout_p=0.0,
        is_causal=is_causal,
        scale=scaling,
    )
    return output.to(input_dtype), None


def get_custom_sdpa_for_ring_kv_cache(
    exportable_module: torch.nn.Module,
) -> Callable:
    # lazy importing to avoid version dependent class definition
    from executorch import version

    try:
        from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
            CustomRingKVCache,
        )
    except ImportError:
        raise ImportError(
            f"CustomRingKVCache not available in version {version.__version__} of ExecuTorch."
        )

    def _custom_sdpa_for_ring_kv_cache(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Union[torch.Tensor, "BlockMask"],  # noqa
        scaling: Optional[float] = None,
        softcap: Optional[float] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        is_sliding = getattr(module, "is_sliding", False)
        if is_sliding:
            # lazy import to avoid being in the optimum import path
            # for et <= 0.6.0 version
            from optimum.executorch.attentions.custom_kv_cache import (
                ETCustomHybridCache,
            )

            layer_idx = module.layer_idx
            assert (
                layer_idx is not None
            ), "layer_idx is not set for sliding window attention."
            hybrid_cache = exportable_module.model.cache
            assert isinstance(
                hybrid_cache, ETCustomHybridCache
            ), f"Expected HybridCache, got {type(hybrid_cache)}"
            ring_cache = hybrid_cache.get_layer_cache(layer_idx)
            assert isinstance(
                ring_cache, CustomRingKVCache
            ), f"Expected CustomRingKVCache, got {type(ring_cache)}"
            input_pos = hybrid_cache.cache_position[0].item()
            seqlen = query.shape[2]
            attention_mask = ring_cache.create_causal_mask_for_ring_buffer(
                input_pos, seqlen
            )
            kwargs.update({"is_sliding": True})
            return custom_sdpa_with_start_pos_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                softcap,
                head_mask,
                **kwargs,
            )
        else:
            return custom_sdpa_with_start_pos_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                softcap,
                head_mask,
                **kwargs,
            )

    return _custom_sdpa_for_ring_kv_cache
