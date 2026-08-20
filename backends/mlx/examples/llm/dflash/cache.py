# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent per-layer KV cache for the DFlash draft model. 

Uses the TorchExportableModuleWithStaticCache pattern with mutable cache buffers and an explicit cache_position. Built on the existing MLX KVCache so cache writes use the MLX KV-cache operator rather than Python slicing. 
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

from executorch.backends.mlx.llm.cache import KVCache


class DFlashDraftKVCache(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_ctx_len: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.max_ctx_len = max_ctx_len
        self.layers = nn.ModuleList(
            [
                KVCache(
                    max_batch_size=1,
                    max_context_length=max_ctx_len,
                    n_heads=num_heads,
                    head_dim=head_dim,
                    enable_dynamic_shape=True,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

    def write(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Union[torch.Tensor, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns the full cache buffer for the layer.
        # The caller narrows it to the valid length before attention.
        return self.layers[layer_idx].update(cache_position, key_states, value_states)

    @staticmethod
    def valid_len_after(cache_position: torch.Tensor) -> torch.Tensor:
        # Returns one past the last position written in the current update.
        # The caller uses this length to exclude unwritten or stale cache entries.
        return cache_position[-1] + 1
