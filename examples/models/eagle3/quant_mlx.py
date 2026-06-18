# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Int4 quantization + MLX packing for the EAGLE-3 draft head.

Mirrors the target's MLX recipe: int4 group-32 linears, int8 per-axis embedding,
norms left in bf16. ``quantize_model`` only captures persistent state, so the
draft's non-persistent ``d2t``/``t2d`` buffers and KV cache are restored here.
"""

import torch

from executorch.examples.models.eagle3.draft import Eagle3Draft
from executorch.examples.models.gemma4_31b.quant import (
    DEFAULT_MLX_PACKERS,
    pack_model,
    QuantConfig,
    quantize_model,
    QuantRecipe,
    QuantRule,
)


def _draft_recipe(hidden_size: int, group_size: int) -> QuantRecipe:
    int4 = QuantConfig(bits=4, group_size=group_size, symmetric=False, method="min_max")
    int8_per_axis = QuantConfig(
        bits=8, group_size=hidden_size, symmetric=True, method="min_max"
    )
    return QuantRecipe(
        rules=[
            QuantRule(r"embed_tokens\.weight", int8_per_axis),
            QuantRule(r".*norm\.weight", None),
            QuantRule(r".*\.weight", int4),
        ]
    )


def quantize_pack_draft_for_mlx(
    draft: Eagle3Draft,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 32,
) -> Eagle3Draft:
    """Return a new MLX-packed int4 draft (linears int4, embedding int8, norms bf16)."""
    config = draft.config
    state = quantize_model(draft, _draft_recipe(config.hidden_size, group_size), dtype)

    packed = Eagle3Draft(config)
    pack_model(packed, state, packers=DEFAULT_MLX_PACKERS)
    # quantize_model skips non-persistent buffers; carry the vocab maps over and
    # allocate the KV cache in the compute dtype.
    packed.register_buffer("d2t", draft.d2t.clone(), persistent=False)
    packed.register_buffer("t2d", draft.t2d.clone(), persistent=False)
    packed.allocate_kv_cache(dtype, device="cpu")
    return packed.eval()
