# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Components for supporting Attention Sink. See
# https://arxiv.org/abs/2309.17453 for more details about Attention Sink.

from typing import Optional

import torch

from executorch.examples.models.llama.llama_transformer import KVCache, ModelArgs
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


class KVCacheWithAttentionSink(KVCache):
    """
    KV cache that supports attention sink. It keeps the initial few tokens as attention sink.
    For other tokens, it uses a sliding window to keep the most recent tokens.

    Parameters:
        window_size: the size of the sliding window
        sink_size: the number of initial tokens to keep as attention sink
        eviction_batch_size: the number of tokens to evict in batch when there is not enough space in the KV cache
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        transpose_cache: bool,
        enable_dynamic_shape: bool,
        rope: RopeWithAttentionSink,
        max_batch_size: int = 1,
        window_size: int = 2044,
        sink_size: int = 4,
        eviction_batch_size: int = 1,
        dtype=torch.float32,
    ):
        super().__init__(
            max_batch_size=max_batch_size,
            max_seq_length=window_size + sink_size,
            n_heads=n_heads,
            head_dim=head_dim,
            transpose_cache=transpose_cache,
            enable_dynamic_shape=enable_dynamic_shape,
            dtype=dtype,
        )
        self.rope = rope
        self.window_size = window_size
        self.sink_size = sink_size
        self.eviction_batch_size = eviction_batch_size
        self.position_shift = 0

    def evict_tokens(self, input_pos: torch.Tensor, seq_len: int) -> int:
        """
        Evict old tokens from the cache to make rooms for new tokens.

        Parameters:
            input_pos: the start position of the incoming token in the actual sequence
            seq_len: the length of the incoming sequence
            rope: the rope object to use for rerotating k

        Returns:
            the number of tokens to evict from the cache which is also the number of
            positions to shift for incoming tokens
        """
        input_pos_item = input_pos.item()
        torch._check_is_size(input_pos_item)
        if input_pos_item + self.position_shift + seq_len > self.max_seq_length:
            # There are not enough spaces in the cache to store the new tokens.
            # We need to evict some old tokens and shift some recent tokens.
            num_to_evict = max(
                input_pos_item + self.position_shift - self.max_seq_length + seq_len,
                self.eviction_batch_size,
            )
            num_to_keep = (
                input_pos_item + self.position_shift - self.sink_size - num_to_evict
            )
            num_empty_space = self.window_size - num_to_keep
            dim_to_slice = 2 if self.transpose_cache else 1
            k_to_keep = self.k_cache.narrow(
                dim_to_slice,
                self.sink_size + num_to_evict,  # pyre-ignore [6]
                num_to_keep,  # pyre-ignore [6]
            )
            if self.transpose_cache:
                k_to_keep = self.rope.rerotate_k(
                    k=k_to_keep.transpose(1, 2),
                    original_position=(  # pyre-ignore [6]
                        self.sink_size + num_to_evict
                    ),
                    new_position=self.sink_size,
                ).transpose(1, 2)
            else:
                k_to_keep = self.rope.rerotate_k(
                    k=k_to_keep,
                    original_position=(  # pyre-ignore [6]
                        self.sink_size + num_to_evict
                    ),
                    new_position=self.sink_size,
                )
            self.k_cache = torch.cat(
                [
                    self.k_cache.narrow(dim_to_slice, 0, self.sink_size),
                    k_to_keep,
                    torch.zeros_like(
                        self.k_cache.narrow(
                            dim_to_slice, 0, num_empty_space  # pyre-ignore [6]
                        )
                    ),
                ],
                dim=dim_to_slice,
            )
            self.v_cache = torch.cat(
                [
                    self.v_cache.narrow(dim_to_slice, 0, self.sink_size),
                    self.v_cache.narrow(
                        dim_to_slice,
                        self.sink_size + num_to_evict,  # pyre-ignore [6]
                        num_to_keep,  # pyre-ignore [6]
                    ),
                    torch.zeros_like(
                        self.v_cache.narrow(
                            dim_to_slice, 0, num_empty_space  # pyre-ignore [6]
                        )
                    ),
                ],
                dim=dim_to_slice,
            )
            self.position_shift -= num_to_evict  # pyre-ignore [8]
        return self.position_shift
