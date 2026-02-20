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
    beyond the sliding window. The actual KV cache size is sink_size + window_size.
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
        # The KV cache size is sink_size + window_size = max_seq_length
        self.kv_cache_size = sink_size + window_size
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
    pos_q = start_pos + torch.arange(seq_len, dtype=torch.long, device=cache_positions.device).view(-1, 1)
    delta = pos_q - cache_positions

    # Valid if position is filled (>= 0) and causal (delta >= 0)
    is_valid = (cache_positions >= 0) & (delta >= 0)

    # Sink tokens (original positions 0 to sink_size-1) are always visible
    is_sink = cache_positions < sink_size

    # Window tokens must be within sliding window
    is_in_window = delta <= window_size

    # Final mask: valid AND (is_sink OR is_in_window)
    attn_mask = is_valid & (is_sink | is_in_window)
    # IMPORTANT: Must use float32 for the mask - C++ SDPA expects ScalarType::Float
    attn_mask = torch.where(attn_mask, torch.tensor(0.0, dtype=torch.float32), torch.tensor(float("-inf"), dtype=torch.float32))
    return attn_mask


class CachePositionsManagerWithSink(nn.Module):
    """
    Manages cache positions for attention sink + sliding window.

    For sink_size=0: behaves exactly like original CachePositionsManager (simple ring buffer).
    For sink_size>0: sink tokens (indices 0 to sink_size-1) are NEVER overwritten.
                     Ring buffer only cycles through indices sink_size to cache_size-1.

    IMPORTANT: cache_size should be the actual cache dimension size (sink_size + window_size).
    """

    def __init__(self, cache_size: int, sink_size: int = 0):
        super().__init__()
        self.max_context_length = cache_size
        self.sink_size = sink_size
        # Ring buffer size = cache_size - sink_size
        self.ring_size = cache_size - sink_size
        # Initialize to -1 to mark unwritten positions
        # The mask uses (cache_positions >= 0) to check if a position is valid
        self.register_buffer(
            "cache_positions",
            torch.full((self.max_context_length,), -1, dtype=torch.long, device="cpu"),
        )

    def calculate_positions_and_update_indices(
        self, input_pos: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        Calculate indices into k_cache, v_cache for placing k_val, v_val.

        Index calculation:
        - Position < sink_size: index = position (sink tokens at fixed indices)
        - Position >= sink_size: index = sink_size + (position - sink_size) % ring_size

        This ensures sink tokens (indices 0 to sink_size-1) are NEVER overwritten.
        """
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)

        # Original positions for the sequence
        orig_positions = torch.arange(seq_len, dtype=torch.long) + start_pos

        if self.sink_size == 0:
            # Simple ring buffer: just mod by cache size
            indices = orig_positions % self.max_context_length
        else:
            # Shifted ring buffer: sink tokens at fixed indices, rest in ring buffer
            # For position >= sink_size: index = sink_size + (position - sink_size) % ring_size
            shifted = orig_positions - self.sink_size
            ring_indices = self.sink_size + (shifted % self.ring_size)
            # For position < sink_size: use position directly
            indices = torch.where(orig_positions < self.sink_size, orig_positions, ring_indices)

        # Update cache_positions to track what position is at each index
        # Only update the indices we're writing to
        self.cache_positions.index_copy_(0, indices, orig_positions)

        return indices


class KVCacheWithAttentionSink(KVCache):
    """
    KV cache that supports attention sink with torch.export compatibility.

    Uses a ring buffer approach for the sliding window portion while keeping
    the first sink_size tokens fixed. This avoids dynamic shape operations.

    Cache layout: [sink: 0 to sink_size-1] [ring_buffer: sink_size to sink_size + window_size - 1]
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
        max_context_len: Optional[int] = None,
        dtype=torch.float32,
    ):
        # Total cache size (KV cache) = sink_size + window_size = max_seq_length
        # max_context_len is for RoPE position encoding limit, NOT cache size
        total_cache_size = sink_size + window_size

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
        self.cache_positions_manager = CachePositionsManagerWithSink(total_cache_size, sink_size)

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
        indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new key-value pairs.
        Uses ring buffer indexing for positions >= sink_size.
        """
        seq_len = k_val.size(2)
        assert seq_len <= self.k_cache.size(
            2
        ), f"Update sequence length({seq_len}) for kv cache must be smaller than the cache size({self.k_cache.size(2)})"

        if indices is None:
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
    **kwargs,
):
    """
    Forward function for attention with attention sink KV cache.
    Uses ring buffer masking for proper attention patterns.
    """
    assert self.use_kv_cache

    input_pos = kwargs.get("input_pos")
    assert input_pos is not None

    # Extract cache_indices if provided (injected by Transformer forward)
    cache_indices = kwargs.get("cache_indices")

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
    k, v = self.kv_cache.update(input_pos, k, v, cache_indices)

    # Use ring buffer mask since we have is_ring_buffer=True
    start_pos = input_pos[0].item()
    torch._check_is_size(start_pos)
    attn_mask = self.kv_cache.create_causal_mask_for_ring_buffer(start_pos, seqlen)

    # For SDPA with attention mask, we pass 0 as start_pos since:
    # 1. The mask handles all masking logic (including ring buffer / attention sink)
    # 2. is_causal=False so start_pos is not used for causal masking
    # 3. This avoids issues with torch.export and data-dependent tensor creation
    # The SDPACustom with use_attention_mask=True will use 0 anyway (see sdpa.py line 62)

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
    max_context_len: int,
):
    for _, child_module in module._modules.items():
        if len(list(child_module.children())) > 0:  # pyre-ignore [16]
            _replace_attention(
                module=child_module,  # pyre-ignore [6]
                rope_with_attention_sink=rope_with_attention_sink,
                sink_size=sink_size,
                window_size=window_size,
                eviction_batch_size=eviction_batch_size,
                max_context_len=max_context_len,
            )

        if isinstance(child_module, AttentionMHA):
            kv_cache = child_module.kv_cache
            kv_cache_with_attention_sink = KVCacheWithAttentionSink(
                n_heads=kv_cache.n_heads,
                head_dim=kv_cache.head_dim,
                enable_dynamic_shape=child_module.enable_dynamic_shape,
                rope=rope_with_attention_sink,
                max_batch_size=kv_cache.max_batch_size,
                window_size=window_size,
                sink_size=sink_size,
                eviction_batch_size=eviction_batch_size,
                max_context_len=max_context_len,
                dtype=kv_cache.k_cache.dtype,
            )
            child_module.kv_cache = kv_cache_with_attention_sink

            # If using SDPACustom (fused SDPA op), enable attention mask support
            # so it uses our ring buffer / attention sink mask instead of simple causal mask
            if "SDPACustom" in child_module.SDPA.__class__.__name__:
                child_module.SDPA.use_attention_mask = True

            # Note: We don't replace the forward method. AttentionMHA.forward
            # already handles is_ring_buffer=True (see attention.py) by:
            # 1. Calling kv_cache.update(input_pos, k, v)
            # 2. Calling kv_cache.create_causal_mask_for_ring_buffer(start_pos, seqlen)
            # This avoids torch.export issues with monkey-patched forward methods.


def enable_attention_sink(
    module: torch.nn.Module,
    params: ModelArgs,
    sink_size: int,
    window_size: int,
    eviction_batch_size: int,
    max_context_len: Optional[int] = None,
) -> torch.nn.Module:
    """
    Transform the model to be able to run inference with Attention Sink.
    There mainly three steps:
    - Replace Rope with RopeWithAttentionSink
    - Replace Attention's KVCache with KVCacheWithAttentionSink
    - Replace Attention's forward with attention_sink_forward
    """
    if max_context_len is None:
        # max_context_len is for RoPE position encoding limit
        # Default to kv_cache_size if not specified, but typically should be larger (e.g., 8192)
        max_context_len = sink_size + window_size

    # We update params.max_context_len for RoPE position encoding limit
    # This ensures the RoPE frequency table is large enough for generation
    params.max_context_len = max_context_len

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
        max_context_len=max_context_len,
    )

    # Add metadata methods for IOManager detection
    # These method names must match the constants in constants.h:
    # kAttentionSinkSize = "attention_sink_size"
    # kAttentionSinkWindowSize = "attention_sink_window_size"
    def attention_sink_size(self):
        return sink_size

    def attention_sink_window_size(self):
        return window_size

    # Bind methods to module
    module.attention_sink_size = types.MethodType(attention_sink_size, module)
    module.attention_sink_window_size = types.MethodType(attention_sink_window_size, module)

    # Note: We do NOT modify get_example_inputs or forward signature.
    # The ring buffer calculates cache indices internally from input_pos,
    # so the model can use the standard 2-input signature (tokens, input_pos).
    # This allows the standard IOManager to work without modification.

    return module
