# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent per-layer KV cache for the DFlash draft model.

Follows the TorchExportableModuleWithStaticCache pattern (per review): cache
tensors are registered as mutable buffers and cache_position is passed
through the call rather than tracked internally. Built on the existing
KVCache (backends/mlx/llm/cache.py), so writes go through
torch.ops.mlx.kv_cache_update instead of a Python slice (avoids the
GuardOnDataDependentSymNode failure hit by an earlier attempt).
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
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList(
            [
                KVCache(
                    max_batch_size=1,
                    max_context_length=max_seq_len,
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
        # Write K/V at cache_position and return the FULL buffer for this
        # layer. Caller masks the unwritten tail via valid_mask(). Extract
        # cache_position once and reuse across layers, not once per layer.
        return self.layers[layer_idx].update(cache_position, key_states, value_states)

    def valid_mask(self, valid_len: Union[torch.Tensor, int], device=None) -> torch.Tensor:
        # True for positions [0, valid_len), False for the unwritten tail.
        # This is the read-side of the DFlash "append and crop" semantics
        # (per review): valid_len is authoritative, so any cache slot at or
        # beyond it -- whether never written, or holding stale data from a
        # position that's since been cropped away -- is masked out of
        # attention regardless of what's physically in the buffer. Combined
        # with the append-only write() above, this gives crop behavior
        # without a data-dependent narrow(): cropping to `valid_len` is just
        # attending with this mask rather than physically truncating the
        # buffer (which would need .item() and break export).
        positions = torch.arange(self.max_seq_len, device=device)
        return positions < valid_len

    @staticmethod
    def valid_len_after(cache_position: torch.Tensor) -> torch.Tensor:
        # The authoritative confirmed-context length after a write at
        # cache_position: one past the last written position. This is the
        # "crop point" -- everything from here on is invalid, and valid_mask()
        # enforces exactly that. A named method so the append-and-crop
        # semantics are explicit rather than an inline `cache_position[-1]+1`.
        return cache_position[-1] + 1

    def reset(self) -> None:
        for layer in self.layers:
            layer.k_cache.zero_()
            layer.v_cache.zero_()
