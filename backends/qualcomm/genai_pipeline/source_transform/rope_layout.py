# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict transforms for RoPE weight layout."""

from __future__ import annotations

from typing import Any, Dict


def permute_partial_rope(
    state_dict: Dict[str, Any],
    *,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    partial_rotary_factor: float,
) -> Dict[str, Any]:
    """change to HF weight to improve the performance of RoPE in HTP backend."""

    def permute(w, heads, partial_rotary_dim):
        dim_0 = w.size(0)
        dim_1 = w.size(1)
        transformed_weight = (
            w.view(heads, -1, dim_0 // heads // 2 // partial_rotary_dim, 2, dim_1)
            .transpose(2, 3)
            .reshape(dim_0, dim_1)
        )
        return transformed_weight

    # TODO: handle cases where input size isn't divisible.
    partial_rotary_dim = int(1 // partial_rotary_factor)
    for layer_i in range(n_layers):
        state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
            state_dict[f"layers.{layer_i}.attention.wq.weight"],
            n_heads,
            partial_rotary_dim,
        )
        state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
            state_dict[f"layers.{layer_i}.attention.wk.weight"],
            n_kv_heads,
            partial_rotary_dim,
        )

    return state_dict
