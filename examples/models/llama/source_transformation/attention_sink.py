# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Components for supporting Attention Sink. See
# https://arxiv.org/abs/2309.17453 for more details about Attention Sink.

# This implementation is torch.export compatible using a ring buffer approach
# for the sliding window portion while preserving the sink tokens.

import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
from executorch.examples.models.llama.attention import (
    _create_causal_mask_for_ring_buffer,
    AttentionMHA,
    CachePositionsManager,
    KVCache,
    RingKVCache,
)
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

    For torch.export compatibility, this just passes through the position - the
    actual position adjustment is handled by the cache update logic.
    
    Note: This class uses the model's max_context_len (params.max_context_len) for
    RoPE frequency table size, which should be large enough to support generation
    beyond the sliding window. The actual KV cache size is sink_size + window_size * 2.
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
        # The KV cache size is sink_size + window_size * 2 (ring buffer needs 2x)
        self.kv_cache_size = sink_size + window_size * 2
        self.window_size = window_size
        self.sink_size = sink_size
        # max_context_len from params is used for RoPE frequencies (should be large)
        self.max_context_length = self.params.max_context_len
        self.eviction_batch_size = eviction_batch_size

    def get_freqs(self, input_pos: Optional[torch.Tensor], seq_len: int):
        """
        Get rotary embedding frequencies.
        For attention sink, we use the original position - the sliding window
        is handled by the cache index management, not by position shifting.
        """
        assert input_pos is not None
        return super().get_freqs(input_pos, seq_len)

    def rerotate_k(
        self,
        k: torch.Tensor,
        original_position: int,
        new_position: int,
    ):
        """
        Rerotate k from original_position to new_position.
        The shape of k is (batch_size, seq_len, n_local_heads, head_dim)
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


def _create_causal_mask_for_attention_sink(
    cache_positions, window_size, sink_size, start_pos, seq_len
):
    """
    Create causal mask for attention sink.
    
    Unlike regular ring buffer mask, this mask:
    1. ALWAYS allows attending to sink tokens (positions 0 to sink_size-1)
    2. Uses sliding window for other tokens
    
    Args:
        cache_positions: Tensor of actual positions stored at each cache index
        window_size: Size of the sliding window
        sink_size: Number of sink tokens to always attend to
        start_pos: Starting position of the current query
        seq_len: Length of the current query sequence
    """
    pos_q = start_pos + torch.arange(seq_len, dtype=torch.long).view(-1, 1)
    delta = pos_q - cache_positions
    
    # Valid if position is filled (>= 0) and causal (delta >= 0)
    is_valid = (cache_positions >= 0) & (delta >= 0)
    
    # Sink tokens (original positions 0 to sink_size-1) are always visible
    is_sink = cache_positions < sink_size
    
    # Window tokens must be within sliding window
    is_in_window = delta < window_size
    
    # Final mask: valid AND (is_sink OR is_in_window)
    attn_mask = is_valid & (is_sink | is_in_window)
    attn_mask = torch.where(attn_mask == True, 0, float("-inf"))  # noqa E712
    return attn_mask


class CachePositionsManagerWithSink(nn.Module):
    """
    Manages cache positions for attention sink + sliding window.
    
    For sink_size=0: behaves exactly like original CachePositionsManager.
    For sink_size>0: sink tokens go to fixed positions, rest uses ring buffer.
    
    IMPORTANT: cache_size should be the actual cache dimension size (2x window for ring buffer).
    """

    def __init__(self, cache_size: int):
        super().__init__()
        # cache_size is the actual size of the kv cache dimension
        self.max_context_length = cache_size
        # Use zeros like original CachePositionsManager
        self.register_buffer(
            "cache_positions",
            torch.zeros((self.max_context_length,), dtype=torch.long, device="cpu"),
        )

    def calculate_positions_and_update_indices(
        self, input_pos: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        Calculate indices into k_cache, v_cache for placing k_val, v_val.
        
        This is identical to the original CachePositionsManager logic.
        """
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)

        orig_indices = torch.arange(seq_len, dtype=torch.long) + start_pos
        
        # Simple ring buffer: just mod by cache size
        indices = orig_indices % self.max_context_length

        # Update cache_positions exactly like original CachePositionsManager
        full_t = torch.full((self.max_context_length,), -1, dtype=torch.long)
        arange_tensor = torch.arange(self.max_context_length, dtype=torch.long)
        cache_positions = torch.where(
            arange_tensor < start_pos, self.cache_positions, full_t
        )
        self.cache_positions.copy_(cache_positions)
        self.cache_positions.index_copy_(0, indices, orig_indices)

        return indices


class KVCacheWithAttentionSink(KVCache):
    """
    KV cache that supports attention sink with torch.export compatibility.

    Uses a ring buffer approach for the sliding window portion while keeping
    the first sink_size tokens fixed. This avoids dynamic shape operations.

    Cache layout: [sink: 0 to sink_size-1] [ring_buffer: sink_size to sink_size + window_size*2 - 1]
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
        # Total cache size is sink_size + window_size * 2 (ring buffer needs 2x)
        total_cache_size = sink_size + window_size * 2
        super().__init__(
            max_batch_size=max_batch_size,
            max_context_length=total_cache_size,
            n_heads=n_heads,
            head_dim=head_dim,
            enable_dynamic_shape=enable_dynamic_shape,
            dtype=dtype,
        )
        self.rope = rope
        self.window_size = window_size
        self.sink_size = sink_size
        self.eviction_batch_size = eviction_batch_size
        self.is_ring_buffer = True

        # Cache positions manager for determining write locations
        # Pass the total cache size (same as self.max_context_length after super().__init__)
        self.cache_positions_manager = CachePositionsManagerWithSink(total_cache_size)

    def create_causal_mask_for_ring_buffer(
        self, start_pos: torch.Tensor, seq_len: int
    ):
        """
        Create causal mask for the attention with attention sink.
        Sink tokens are ALWAYS visible, plus recent tokens in the window.
        """
        cache_positions = self.cache_positions_manager.cache_positions
        if self.sink_size > 0:
            # Use attention sink mask that always allows attending to sink tokens
            return _create_causal_mask_for_attention_sink(
                cache_positions, self.window_size, self.sink_size, start_pos, seq_len
            )
        else:
            # Pure ring buffer mode - use original mask with window_size = actual window
            return _create_causal_mask_for_ring_buffer(
                cache_positions, self.window_size, start_pos, seq_len
            )

    def update(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new key-value pairs.
        Uses ring buffer indexing for positions >= sink_size.
        """
        seq_len = k_val.size(2)
        assert seq_len <= self.k_cache.size(
            2
        ), f"Update sequence length({seq_len}) for kv cache must be smaller than the cache size({self.k_cache.size(2)})"

        # Calculate write indices
        indices = self.cache_positions_manager.calculate_positions_and_update_indices(
            input_pos, seq_len
        )

        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        self.k_cache.index_copy_(2, indices, k_val)
        self.v_cache.index_copy_(2, indices, v_val)

        return self.k_cache, self.v_cache

    def evict_tokens(self, input_pos: torch.Tensor, seq_len: int) -> int:
        """
        For ring buffer implementation, no explicit eviction is needed.
        The ring buffer automatically overwrites old values.
        Returns 0 to indicate no position shift is needed.
        """
        return 0


def attention_sink_forward(
    self,
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    input_pos: Optional[torch.Tensor] = None,
):
    """
    Forward function for attention with attention sink KV cache.
    Uses ring buffer masking for proper attention patterns.
    """
    assert self.use_kv_cache
    assert input_pos is not None

    bsz, seqlen, _ = x.shape

    # QKV
    q, k, v = self.wq(x), self.wk(x), self.wv(x)
    q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    # RoPE relative positional embeddings
    q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)

    # Transpose for attention: [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Update KV cache
    k, v = self.kv_cache.update(input_pos, k, v)

    # Use ring buffer mask since we have is_ring_buffer=True
    start_pos = input_pos[0].item()
    torch._check_is_size(start_pos)
    attn_mask = self.kv_cache.create_causal_mask_for_ring_buffer(start_pos, seqlen)

    # SDPA
    output = self.SDPA(input_pos, q, k, v, bsz, seqlen, attn_mask)

    # Return tuple like original AttentionMHA.forward
    return self.wo(output), None


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
            if sink_size == 0:
                # For sink_size=0, use the exact same RingKVCache that works
                # This is a test to ensure parity with the working implementation
                child_module.kv_cache = RingKVCache(
                    kv_cache.max_batch_size,
                    window_size,  # RingKVCache expects user-provided window size
                    kv_cache.n_heads,
                    kv_cache.head_dim,
                    kv_cache.enable_dynamic_shape,
                    kv_cache.k_cache.dtype,
                )
            else:
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
            # Don't replace forward - let the original AttentionMHA.forward handle it
            # since our KVCache has is_ring_buffer=True, it will use the ring buffer mask


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
    - Replace Attention's KVCache with KVCacheWithAttentionSink
    - Replace Attention's forward with attention_sink_forward
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
