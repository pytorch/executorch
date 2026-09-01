# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Components for supporting Attention Sink. See
# https://arxiv.org/abs/2309.17453 for more details about Attention Sink.

# This implementation is torch.export compatible using a ring buffer approach
# for the sliding window portion while preserving the sink tokens.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from executorch.examples.models.llama.attention import (
    _create_causal_mask_for_ring_buffer,
    _get_ring_cache_size,
    AttentionMHA,
    KVCache,
    RingKVCache,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter


def _get_attention_sink_cache_size(
    max_context_length: int,
    window_size: int,
    sink_size: int,
    max_seq_len: Optional[int] = None,
) -> int:
    """Size a sink cache for fixed sinks, one window, and one input chunk."""
    assert sink_size >= 0, "Attention sink size must be non-negative"
    assert window_size > 0, "Sliding-window size must be positive"
    if max_seq_len is None:
        max_seq_len = max_context_length
    assert max_seq_len > 0, "Maximum sequence length must be positive"
    assert sink_size + window_size <= max_context_length, (
        f"Attention sink size ({sink_size}) plus sliding-window size "
        f"({window_size}) cannot exceed the full context length "
        f"({max_context_length})"
    )
    return sink_size + _get_ring_cache_size(
        max_context_length - sink_size,
        window_size,
        max_seq_len,
    )


class RopeWithAttentionSink(Rope):
    """
    Rope subclass for Attention Sink models.

    Remaps input positions using modular arithmetic so RoPE frequencies stay
    within the cache size bounds, enabling generation beyond max_context_len.

    Position mapping:
      - Sink tokens (pos < sink_size): position preserved as-is
      - Window tokens (pos >= sink_size): wrapped into ring buffer range
        [sink_size, sink_size + ring_size) via modulo

    The ring buffer holds one retained window plus one maximum-size input
    chunk. It is larger than the live window for write-ahead headroom, not to
    keep the live window contiguous -- it can span a wrap. Across a wrap two
    positions remap to a difference that is not their true distance, so RoPE
    preserves relative distance only within a wrap.
    """

    def __init__(
        self,
        params: ModelArgs,
        window_size: int,
        sink_size: int,
    ):
        super().__init__(params)
        self.window_size = window_size
        self.sink_size = sink_size
        self.max_seq_len = (
            params.max_context_len
            if getattr(params, "max_seq_len", None) is None
            else int(params.max_seq_len)
        )
        cache_size = _get_attention_sink_cache_size(
            params.max_context_len,
            window_size,
            sink_size,
            self.max_seq_len,
        )
        self.ring_size = cache_size - sink_size

    def _remap_input_pos(self, input_pos: torch.Tensor) -> torch.Tensor:
        """Remap positions: sink tokens stay, window tokens wrap in ring buffer."""
        return torch.where(
            input_pos < self.sink_size,
            input_pos,
            self.sink_size + (input_pos - self.sink_size) % self.ring_size,
        )

    def get_freqs(self, input_pos: Optional[torch.Tensor], seq_len: int):
        """
        Get rotary embedding frequencies with position remapping.

        For dynamic shape mode input_pos is a single start position, expanded
        here into the full chunk; for static shape mode it already is the full
        position tensor. Either way every position is remapped and indexed.
        Remapping only the start and slicing from there would branch on a
        data-dependent value, which blocks export, and would read past the ring
        on a chunk that wraps.
        """
        assert input_pos is not None
        if not self.params.use_kv_cache:
            return self.freqs_cos[:seq_len], self.freqs_sin[:seq_len]

        if self.params.enable_dynamic_shape:
            # Dynamic shape: input_pos is [start_pos], expand to the whole chunk
            input_pos = input_pos[-1] + torch.arange(
                seq_len, device=input_pos.device, dtype=input_pos.dtype
            )

        # Remap the full position tensor and index
        remapped = self._remap_input_pos(input_pos)
        freqs_cos = self.freqs_cos[remapped]
        freqs_sin = self.freqs_sin[remapped]

        return freqs_cos, freqs_sin


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

    IMPORTANT: cache_size is the actual cache dimension, including sink slots.
    """

    def __init__(self, cache_size: int, sink_size: int = 0):
        super().__init__()
        assert (
            cache_size > sink_size
        ), f"cache_size ({cache_size}) must be larger than sink_size ({sink_size})"
        # cache_size is the actual size of the kv cache dimension
        self.max_context_length = cache_size
        self.sink_size = sink_size
        self.ring_size = cache_size - sink_size
        # Initialize to -1 to indicate empty/unfilled slots
        self.register_buffer(
            "cache_positions",
            torch.full((self.max_context_length,), -1, dtype=torch.long, device="cpu"),
        )

    def calculate_positions_and_update_indices(
        self, input_pos: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """
        Calculate indices into k_cache, v_cache for placing k_val, v_val.

        Sink tokens (positions < sink_size) map to cache slots [0, sink_size).
        Window tokens (positions >= sink_size) use ring buffer in [sink_size, cache_size).
        """
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)

        orig_indices = torch.arange(seq_len, dtype=torch.long) + start_pos

        # Sink tokens go to fixed slots; window tokens use ring buffer
        indices = torch.where(
            orig_indices < self.sink_size,
            orig_indices,
            self.sink_size + (orig_indices - self.sink_size) % self.ring_size,
        )

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

    Cache layout: [fixed sink tokens] [ring buffer for one window + one chunk]
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        enable_dynamic_shape: bool,
        rope: RopeWithAttentionSink,
        window_size: int,
        sink_size: int,
        max_context_length: int,
        max_seq_len: Optional[int] = None,
        max_batch_size: int = 1,
        dtype=torch.float32,
    ):
        self.full_context_length = max_context_length
        self.max_seq_len = (
            max_context_length if max_seq_len is None else int(max_seq_len)
        )
        # Keep the fixed sinks separate from the ring space needed for the
        # retained window and the largest in-flight input chunk.
        total_cache_size = _get_attention_sink_cache_size(
            max_context_length,
            window_size,
            sink_size,
            self.max_seq_len,
        )
        assert rope.ring_size == total_cache_size - sink_size, (
            f"RoPE ring size ({rope.ring_size}) must match the KV-cache ring "
            f"size ({total_cache_size - sink_size})"
        )
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
        self.is_ring_buffer = True

        # Cache positions manager for determining write locations
        # Pass the total cache size (same as self.max_context_length after super().__init__)
        self.cache_positions_manager = CachePositionsManagerWithSink(
            total_cache_size, sink_size
        )

    def create_causal_mask_for_ring_buffer(self, start_pos: int, seq_len: int):
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
        # Verify that window tokens (those mapping to the ring buffer) don't
        # exceed ring_size, which would cause duplicate indices in index_copy_.
        # Sink tokens (positions < sink_size) map to fixed slots and are safe.
        start_pos = input_pos[0].item()
        num_sink_tokens = max(0, min(seq_len, self.sink_size - start_pos))
        num_window_tokens = seq_len - num_sink_tokens
        assert num_window_tokens <= self.cache_positions_manager.ring_size, (
            f"Window tokens ({num_window_tokens}) exceed ring buffer capacity "
            f"({self.cache_positions_manager.ring_size}), which would cause "
            f"non-deterministic behavior with index_copy_"
        )

        # Calculate write indices
        indices = self.cache_positions_manager.calculate_positions_and_update_indices(
            input_pos, seq_len
        )

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
    max_seq_len: Optional[int],
):
    for _, child_module in module._modules.items():
        if len(list(child_module.children())) > 0:  # pyre-ignore [16]
            _replace_attention(
                module=child_module,  # pyre-ignore [6]
                rope_with_attention_sink=rope_with_attention_sink,
                sink_size=sink_size,
                window_size=window_size,
                max_seq_len=max_seq_len,
            )

        if isinstance(child_module, AttentionMHA):
            kv_cache = child_module.kv_cache
            if sink_size == 0:
                # No sink tokens needed — use standard RingKVCache directly
                child_module.kv_cache = RingKVCache(
                    kv_cache.max_batch_size,
                    kv_cache.max_context_length,
                    kv_cache.n_heads,
                    kv_cache.head_dim,
                    kv_cache.enable_dynamic_shape,
                    kv_cache.k_cache.dtype,
                    window_size=window_size,
                    max_seq_len=max_seq_len,
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
                    max_context_length=kv_cache.max_context_length,
                    max_seq_len=max_seq_len,
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
) -> torch.nn.Module:
    """
    Transform the model to be able to run inference with Attention Sink.
    There mainly two steps:
    - Replace Rope with RopeWithAttentionSink
    - Replace Attention's KVCache with KVCacheWithAttentionSink
    """
    rope_with_attention_sink = RopeWithAttentionSink(
        params=params,
        window_size=window_size,
        sink_size=sink_size,
    )
    max_seq_len = getattr(params, "max_seq_len", None)
    _replace_rope(module, rope_with_attention_sink)
    _replace_attention(
        module=module,
        rope_with_attention_sink=rope_with_attention_sink,
        sink_size=sink_size,
        window_size=window_size,
        max_seq_len=max_seq_len,
    )
    return module
