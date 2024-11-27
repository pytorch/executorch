# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Components for supporting Attention Sink. See
# https://arxiv.org/abs/2309.17453 for more details about Attention Sink.

from typing import Optional

import torch

from executorch.examples.models.llama.llama_transformer import ModelArgs, Rope
from executorch.examples.models.llama.rope import (
    apply_rotary_emb_to_k,
    hf_apply_rotary_emb_to_k,
)


class RopeWithAttentionSink(Rope):
    """
    Rope that helps adjust position encoding when tokens are shifted in KVCache.
    For AttentionSink, when tokens are shifted in KVCache, we need to use positions
    in KVCache instead of positions in the actual text.
    """

    def __init__(
        self,
        params: ModelArgs,
        window_size: int,
        sink_size: int,
        eviction_batch_size: int,
    ):
        super().__init__(params)
        if self.params.use_hf_rope:
            self.apply_rotary_emb_to_k = hf_apply_rotary_emb_to_k
        else:
            self.apply_rotary_emb_to_k = apply_rotary_emb_to_k
        self.max_seq_length = window_size + sink_size
        assert self.max_seq_length == self.params.max_seq_len
        self.eviction_batch_size = eviction_batch_size
        self.position_shift = 0

    def get_freqs(self, input_pos: Optional[torch.Tensor], seq_len: int):
        assert input_pos is not None

        input_pos_item = input_pos.item()
        torch._check_is_size(input_pos_item)
        if input_pos_item + self.position_shift + seq_len > self.max_seq_length:
            # There are not enough spaces in the cache to store the new tokens.
            # We need to evict some old tokens and shift some recent tokens.
            num_to_evict = max(
                input_pos_item + self.position_shift - self.max_seq_length + seq_len,
                self.eviction_batch_size,
            )
            self.position_shift -= num_to_evict  # pyre-ignore [8]
        return super().get_freqs(input_pos + self.position_shift, seq_len)

    def rerotate_k(
        self,
        k: torch.Tensor,
        original_position: int,
        new_position: int,
    ):
        """
        Rerotate k from original_position to new_position. This is done by rerotating
        k with (new_position * theta - original_position * theta) with the following matrix:
        (cos(delta), -sin(delta)
         sin(delta), cos(delta))
         where delta = new_position * theta - original_position * theta

         The shape of k is (batch_size, seq_len, n_local_heads, head_dim)

         Based on https://github.com/huggingface/transformers/blame/main/src/transformers/cache_utils.py#L961
        """
        seq_len = k.shape[1]
        original_freqs_cos = self.freqs_cos.narrow(0, original_position, seq_len)
        original_freqs_sin = self.freqs_sin.narrow(0, original_position, seq_len)
        new_freqs_cos = self.freqs_cos.narrow(0, new_position, seq_len)
        new_freqs_sin = self.freqs_sin.narrow(0, new_position, seq_len)
        rerotation_cos = (
            new_freqs_cos * original_freqs_cos + new_freqs_sin * original_freqs_sin
        )
        rerotation_sin = (
            new_freqs_sin * original_freqs_cos - new_freqs_cos * original_freqs_sin
        )

        return self.apply_rotary_emb_to_k(k, rerotation_cos, rerotation_sin)
