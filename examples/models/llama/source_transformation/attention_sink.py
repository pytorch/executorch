# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Components for supporting Attention Sink. See
# https://arxiv.org/abs/2309.17453 for more details about Attention Sink.

import types
from typing import Optional

import torch

from executorch.examples.models.llama.attention import AttentionMHA, KVCache
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import (
    apply_rotary_emb_to_k,
    hf_apply_rotary_emb_to_k,
    Rope,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter


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
        self.max_context_length = window_size + sink_size
        assert self.max_context_length == self.params.max_context_len
        self.eviction_batch_size = eviction_batch_size
        self.register_buffer("position_shift", torch.tensor(0, dtype=torch.int64))

    def get_freqs(self, input_pos: Optional[torch.Tensor], seq_len: int):
        assert input_pos is not None

        def true_fn(position_shift, input_pos, seq_len, max_context_length):
            # There are not enough spaces in the cache to store the new tokens.
            # We need to evict some old tokens and shift some recent tokens.
            num_to_evict = torch.clamp(
                input_pos + position_shift - max_context_length + seq_len,
                min=self.eviction_batch_size,
            )
            position_shift = position_shift - num_to_evict
            return position_shift

        def false_fn(position_shift, input_pos, seq_len, max_context_length):
            return position_shift

        if self.params.enable_dynamic_shape:
            input_pos_scalar = input_pos[0]
        else:
            input_pos_scalar = input_pos

        self.position_shift = torch.cond(
            input_pos_scalar + self.position_shift + seq_len > self.max_context_length,
            true_fn,
            false_fn,
            [
                self.position_shift,
                input_pos_scalar,
                seq_len,
                self.max_context_length,
            ],
        )
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
        enable_dynamic_shape: bool,
        rope: RopeWithAttentionSink,
        window_size: int,
        sink_size: int,
        eviction_batch_size: int,
        max_batch_size: int = 1,
        dtype=torch.float32,
    ):
        super().__init__(
            max_batch_size=max_batch_size,
            max_context_length=window_size + sink_size,
            n_heads=n_heads,
            head_dim=head_dim,
            enable_dynamic_shape=enable_dynamic_shape,
            dtype=dtype,
        )
        self.rope = rope
        self.window_size = window_size
        self.sink_size = sink_size
        self.eviction_batch_size = eviction_batch_size
        self.register_buffer("position_shift", torch.tensor(0, dtype=torch.int64))

    def evict_tokens(self, input_pos: torch.Tensor, seq_len: int) -> torch.Tensor:
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

        def true_fn(
            k_cache, v_cache, position_shift, input_pos, seq_len, max_context_length
        ):
            # There are not enough spaces in the cache to store the new tokens.
            # We need to evict some old tokens and shift some recent tokens.
            num_to_evict = torch.clamp(
                input_pos + position_shift - max_context_length + seq_len,
                min=self.eviction_batch_size,
            )
            num_to_keep = (
                input_pos + position_shift - self.sink_size - num_to_evict
            )
            num_empty_space = self.window_size - num_to_keep
            dim_to_slice = 2
            k_to_keep = k_cache.narrow(
                dim_to_slice,
                self.sink_size + num_to_evict,  # pyre-ignore [6]
                num_to_keep,  # pyre-ignore [6]
            )
            k_to_keep = self.rope.rerotate_k(
                k=k_to_keep.transpose(1, 2),
                original_position=(self.sink_size + num_to_evict),  # pyre-ignore [6]
                new_position=self.sink_size,
            ).transpose(1, 2)
            k_cache = torch.cat(
                [
                    k_cache.narrow(dim_to_slice, 0, self.sink_size),
                    k_to_keep,
                    torch.zeros_like(
                        k_cache.narrow(
                            dim_to_slice, 0, num_empty_space  # pyre-ignore [6]
                        )
                    ),
                ],
                dim=dim_to_slice,
            )
            v_cache = torch.cat(
                [
                    v_cache.narrow(dim_to_slice, 0, self.sink_size),
                    v_cache.narrow(
                        dim_to_slice,
                        self.sink_size + num_to_evict,  # pyre-ignore [6]
                        num_to_keep,  # pyre-ignore [6]
                    ),
                    torch.zeros_like(
                        v_cache.narrow(
                            dim_to_slice, 0, num_empty_space  # pyre-ignore [6]
                        )
                    ),
                ],
                dim=dim_to_slice,
            )
            position_shift = position_shift - num_to_evict
            return k_cache, v_cache, position_shift

        def false_fn(
            k_cache, v_cache, position_shift, input_pos, seq_len, max_context_length
        ):
            return k_cache, v_cache, position_shift

        if self.enable_dynamic_shape:
            input_pos_scalar = input_pos[0]
        else:
            input_pos_scalar = input_pos

        self.k_cache, self.v_cache, self.position_shift = torch.cond(
            input_pos_scalar + self.position_shift + seq_len > self.max_context_length,
            true_fn,
            false_fn,
            [
                self.k_cache,
                self.v_cache,
                self.position_shift,
                input_pos_scalar,
                seq_len,
                self.max_context_length,
            ],
        )

        return self.position_shift


def attention_sink_forward(
    self,
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    input_pos: Optional[torch.Tensor] = None,
):
    assert self.use_kv_cache
    assert input_pos is not None

    bsz, seqlen, _ = x.shape

    # QKV
    q, k, v = self.wq(x), self.wk(x), self.wv(x)
    # We need view_copy elimination
    q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    # Prepare for space in KV cache and get position shift
    position_shift = self.kv_cache.evict_tokens(input_pos, seqlen)

    # RoPE relative positional embeddings with shifted position in KV cache
    q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)

    output = self.SDPA(input_pos + position_shift, q, k, v, bsz, seqlen, self.mask)
    return self.wo(output)


def _replace_rope(
    module: torch.nn.Module, rope_with_attention_sink: RopeWithAttentionSink
):
    def filter_fn(child: torch.nn.Module, cur_fqn: str) -> bool:
        return isinstance(child, Rope)

    def replacement_fn(child: torch.nn.Module) -> torch.nn.Module:
        return rope_with_attention_sink

    _replace_with_custom_fn_if_matches_filter(module, replacement_fn, filter_fn)


def _replace_attention(
    module: torch.nn.Module,
    rope_with_attention_sink: RopeWithAttentionSink,
    sink_size: int,
    window_size: int,
    eviction_batch_size: int,
):
    for _, child_module in module._modules.items():
        if len(list(child_module.children())) > 0:  # pyre-ignore [16]
            _replace_attention(
                module=child_module,  # pyre-ignore [6]
                rope_with_attention_sink=rope_with_attention_sink,
                sink_size=sink_size,
                window_size=window_size,
                eviction_batch_size=eviction_batch_size,
            )

        if isinstance(child_module, AttentionMHA):
            kv_cache = child_module.kv_cache
            kv_cache_with_attention_sink = KVCacheWithAttentionSink(
                n_heads=kv_cache.n_heads,
                head_dim=kv_cache.head_dim,
                enable_dynamic_shape=kv_cache.enable_dynamic_shape,
                rope=rope_with_attention_sink,
                max_batch_size=kv_cache.max_batch_size,
                window_size=window_size,
                sink_size=sink_size,
                eviction_batch_size=eviction_batch_size,
                dtype=kv_cache.k_cache.dtype,
            )
            child_module.kv_cache = kv_cache_with_attention_sink
            child_module.forward = types.MethodType(  # pyre-ignore
                attention_sink_forward, child_module
            )


def enable_attention_sink(
    module: torch.nn.Module,
    params: ModelArgs,
    sink_size: int,
    window_size: int,
    eviction_batch_size: int,
) -> torch.nn.Module:
    """
    Transform the model to be able to run inference with Attention Sink.
    There mainly three steps:
    - Replace Rope with RopeWithAttentionSink
    - Replace Attention's KVCache with KVCacheWithAttentionSink, forward with attention_sink_forward
    """
    rope_with_attention_sink = RopeWithAttentionSink(
        params=params,
        window_size=window_size,
        sink_size=sink_size,
        eviction_batch_size=eviction_batch_size,
    )
    _replace_rope(module, rope_with_attention_sink)
    _replace_attention(
        module=module,
        rope_with_attention_sink=rope_with_attention_sink,
        sink_size=sink_size,
        window_size=window_size,
        eviction_batch_size=eviction_batch_size,
    )
    return module
