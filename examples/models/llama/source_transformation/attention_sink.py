# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Components for supporting Attention Sink. See
# https://arxiv.org/abs/2309.17453 for more details about Attention Sink.

from typing import Tuple

import torch

from executorch.examples.models.llama.llama_transformer import Rope

from torch import nn


class RopeWithAttentionSink(nn.Module):
    """
    Rope that helps adjust position encoding when tokens are shifted in KVCache.
    For AttentionSink, when tokens are shifted in KVCache, we need to use positions
    in KVCache instead of positions in the actual text.
    """

    def __init__(self, rope: Rope):
        super().__init__()
        self.rope = rope

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        original_position: int,
        new_position: int,
        seq_len: int,
    ):
        """
        Rerotate q and k from original_position to new_position. This is done by rerotating q
        and k with (new_position * theta - original_position * theta) with the following matrix:
        (cos(delta), -sin(delta)
         sin(delta), cos(delta))
         where delta = new_position * theta - original_position * theta

         Based on https://github.com/huggingface/transformers/blame/main/src/transformers/cache_utils.py#L961
        """
        original_freqs_cos = self.rope.freqs_cos.narrow(0, original_position, seq_len)
        original_freqs_sin = self.rope.freqs_sin.narrow(0, original_position, seq_len)
        new_freqs_cos = self.rope.freqs_cos.narrow(0, new_position, seq_len)
        new_freqs_sin = self.rope.freqs_sin.narrow(0, new_position, seq_len)
        rerotation_cos = (
            new_freqs_cos * original_freqs_cos + new_freqs_sin * original_freqs_sin
        )
        rerotation_sin = (
            new_freqs_sin * original_freqs_cos - new_freqs_cos * original_freqs_sin
        )

        q, k = self.rope.apply_rotary_emb(q, k, rerotation_cos, rerotation_sin)
        return q, k


class KVCacheWithAttentionSink(nn.Module):
    """
    KV cache that supports attention sink. It keeps the initial few tokens as attention sink.
    For other tokens, it uses a sliding window to keep the most recent tokens.

    Parameters:
        window_size: the size of the sliding window
        sink_size: the number of initial tokens to keep as attention sink
    """

    def __init__(
        self,
        max_batch_size: int,
        window_size: int,
        sink_size: int,
        n_heads: int,
        head_dim: int,
        transpose_cache: bool,
        dtype=torch.float32,
    ):
        super().__init__()
        self.window_size = window_size
        self.sink_size = sink_size
        self.cache_size = window_size + sink_size
        self.is_transposed = transpose_cache
        if transpose_cache:
            cache_shape = (max_batch_size, n_heads, self.cache_size, head_dim)
        else:
            cache_shape = (max_batch_size, self.cache_size, n_heads, head_dim)

        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.transpose_cache = transpose_cache
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        dim_to_slice = 2 if self.transpose_cache else 1
        seq_length = k_val.size(dim_to_slice)

        if start_pos + seq_length <= self.cache_size:
            # There are still enough spaces in the cache to store the new tokens.
            # No need to shift the existing tokens.
            # pyre-ignore: Incompatible parameter type [6]
            narrowed_k = self.k_cache.narrow(dim_to_slice, start_pos, seq_length)
            # pyre-ignore: Incompatible parameter type [6]
            narrowed_v = self.v_cache.narrow(dim_to_slice, start_pos, seq_length)

            narrowed_k.copy_(k_val)
            narrowed_v.copy_(v_val)
        else:
            # There are not enough spaces in the cache to store the new tokens.
            # We need to shift the existing tokens.
            num_to_evict = min(start_pos + seq_length - self.cache_size, seq_length)

            # Shift the existing entries to the left
            k_to_keep = self.k_cache.narrow(
                dim_to_slice,
                self.sink_size + num_to_evict,  # pyre-ignore [6]
                self.window_size - num_to_evict,  # pyre-ignore [6]
            ).clone()
            v_to_keep = self.v_cache.narrow(
                dim_to_slice,
                self.sink_size + num_to_evict,  # pyre-ignore [6]
                self.window_size - num_to_evict,  # pyre-ignore [6]
            ).clone()
            k_new_position = self.k_cache.narrow(
                dim_to_slice,
                self.sink_size,
                self.window_size - num_to_evict,  # pyre-ignore [6]
            )
            v_new_position = self.v_cache.narrow(
                dim_to_slice,
                self.sink_size,
                self.window_size - num_to_evict,  # pyre-ignore [6]
            )
            k_new_position.copy_(k_to_keep)
            v_new_position.copy_(v_to_keep)

            # Appending new entries
            narrowed_k = self.k_cache.narrow(
                dim_to_slice, self.cache_size - seq_length, seq_length
            )
            narrowed_v = self.v_cache.narrow(
                dim_to_slice, self.cache_size - seq_length, seq_length
            )
            narrowed_k.copy_(k_val)
            narrowed_v.copy_(v_val)
        return self.k_cache, self.v_cache
