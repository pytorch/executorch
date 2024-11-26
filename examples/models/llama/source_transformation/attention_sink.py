# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Components for supporting Attention Sink. See
# https://arxiv.org/abs/2309.17453 for more details about Attention Sink.

from typing import Optional

import torch

from executorch.examples.models.llama.llama_transformer import ModelArgs
from executorch.examples.models.llama.rope import (
    apply_rotary_emb_to_k,
    hf_apply_rotary_emb,
    hf_apply_rotary_emb_to_k,
    RotaryEmbedding,
)


class RopeWithAttentionSink(torch.nn.Module):
    """
    Rope that helps adjust position encoding when tokens are shifted in KVCache.
    For AttentionSink, when tokens are shifted in KVCache, we need to use positions
    in KVCache instead of positions in the actual text.
    """

    def __init__(self, params: ModelArgs, freqs_cos, freqs_sin):
        super().__init__()
        self.params = params
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin
        if self.params.use_hf_rope:
            self.apply_rotary_emb = hf_apply_rotary_emb
            self.apply_rotary_emb_to_k = hf_apply_rotary_emb_to_k
        else:
            self.apply_rotary_emb = RotaryEmbedding()
            self.apply_rotary_emb_to_k = apply_rotary_emb_to_k

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ):
        seq_len = q.shape[1]
        if self.params.use_kv_cache:
            assert (
                input_pos is not None
            ), "input_pos must be provided when use_kv_cache is True"

            if self.params.enable_dynamic_shape:
                # when KV cache is used, seqlen is most likely 1. We want to slice from the start_pos.
                input_pos_item = input_pos[-1].item()
                torch._check_is_size(input_pos_item)
                torch._check(input_pos_item < self.params.max_seq_len)
                # pyre-ignore: Incompatible parameter type [6]: torch.narrow does expect int or Tensor
                freqs_cos = self.freqs_cos.narrow(0, input_pos_item, seq_len)
                # pyre-ignore: Incompatible parameter type [6]
                freqs_sin = self.freqs_sin.narrow(0, input_pos_item, seq_len)
            else:
                # When not using dynamic shape, use of the .item results in
                # symints, due to querying the data from tensor.
                # this path avoids that for mps backend, although probably mps backend
                # can support dynamic shape?
                assert (
                    seq_len == 1
                ), "Expected seq_len to be 1 when using kv_cache without dynamic shape"
                freqs_cos = self.freqs_cos[input_pos]
                freqs_sin = self.freqs_sin[input_pos]

        else:
            assert input_pos is None, "input_pos is unused when use_kv_cache is False"
            freqs_cos = self.freqs_cos[:seq_len]
            freqs_sin = self.freqs_sin[:seq_len]
        q, k = self.apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        return q, k

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
