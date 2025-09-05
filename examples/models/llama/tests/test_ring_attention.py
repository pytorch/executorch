# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from enum import Enum

import torch
from executorch.examples.models.llama.attention import AttentionMHA, RingKVCache
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    CustomKVCache,
    CustomRingKVCache,
    QuantizedKVCache,
    QuantizedRingKVCache,
    replace_kv_cache_with_custom_kv_cache,
    replace_kv_cache_with_quantized_kv_cache,
)
from torch.nn.attention import SDPBackend


class KVCacheType(Enum):
    REGULAR = "regular"
    QUANTIZED = "quantized"
    CUSTOM = "custom"


class TestRingAttention(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.batch_size = 1
        self.seq_len = 1  # Single token processing
        self.dim = 64
        self.n_heads = 4
        self.n_kv_heads = 4
        self.head_dim = 16
        self.max_context_len = 16
        self.sliding_window = 8
        self.dtype = torch.float32
        self.device = "cpu"

    def _create_baseline_attention(
        self, seq_len: int, kv_cache_type: KVCacheType = KVCacheType.REGULAR
    ):
        """Create baseline attention with regular KV cache."""
        # Create model args
        self.args = ModelArgs(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_batch_size=self.batch_size,
            max_context_len=self.max_context_len,
            use_kv_cache=True,
            enable_dynamic_shape=True,
        )

        # Create RoPE instance
        self.rope = Rope(self.args)

        attention = AttentionMHA(self.args, layer_id=0, rope=self.rope)
        attention.mask = self._create_sliding_window_mask(
            seq_len, self.max_context_len, self.sliding_window
        )

        # Replace the KV cache with the specified type
        if kv_cache_type == KVCacheType.QUANTIZED:
            # Create a copy to avoid modifying the original attention
            attention_copy = copy.deepcopy(attention)
            # Replace KVCache with QuantizedKVCache
            replace_kv_cache_with_quantized_kv_cache(attention_copy)
            return attention_copy
        elif kv_cache_type == KVCacheType.CUSTOM:
            # Create a copy to avoid modifying the original attention
            attention_copy = copy.deepcopy(attention)
            # Replace KVCache with CustomKVCache
            replace_kv_cache_with_custom_kv_cache(attention_copy)
            return attention_copy
        else:
            return attention

    def _create_ring_attention(
        self, attention, kv_cache_type: KVCacheType = KVCacheType.REGULAR
    ):
        """Create attention with ring buffer KV cache."""
        assert self.sliding_window is not None
        # Create RoPE instance
        self.rope = Rope(self.args)
        baseline_attention = copy.deepcopy(attention)

        # Replace the KV cache with a ring buffer KV cache based on the type
        if isinstance(baseline_attention.kv_cache, QuantizedKVCache):
            # Replace QuantizedKVCache with QuantizedRingKVCache
            baseline_attention.kv_cache = QuantizedRingKVCache.from_quantized_kv_cache(
                baseline_attention.kv_cache,
                self.sliding_window,
            )
        elif isinstance(baseline_attention.kv_cache, CustomKVCache):
            # Replace CustomKVCache with CustomRingKVCache
            baseline_attention.kv_cache = CustomRingKVCache.from_custom_kv_cache(
                baseline_attention.kv_cache,
                self.sliding_window,
            )
        else:
            # Replace regular KVCache with RingKVCache
            baseline_attention.kv_cache = RingKVCache(
                self.args.max_batch_size,
                self.sliding_window,
                self.n_kv_heads,
                self.head_dim,
                self.args.enable_dynamic_shape,
                self.dtype,
            )
        return baseline_attention

    def _create_sliding_window_mask(self, seq_len, context_len, window_size):
        """Create a sliding window mask for the baseline."""
        mask = torch.full((seq_len, context_len), float("-inf"), dtype=self.dtype)
        for i in range(seq_len):
            pos = i
            # Allow attention to window_size previous positions
            start_idx = max(0, pos - window_size + 1)
            mask[i, start_idx : pos + 1] = 0
        return mask

    def _run_test_with_kv_cache_type(self, test_func, kv_cache_type: KVCacheType):
        """Run a test with the specified KV cache type."""
        original_test_name = test_func.__name__
        print(f"\nRunning {original_test_name} with {kv_cache_type.value} KV cache")
        test_func(kv_cache_type)

    def test_single_token_processing(
        self, kv_cache_type: KVCacheType = KVCacheType.REGULAR
    ):
        """Test that ring buffer and baseline produce the same output for single token processing."""
        seq_len = 10
        self.sliding_window = 4
        baseline_attn = self._create_baseline_attention(seq_len, kv_cache_type)
        ring_attn = self._create_ring_attention(baseline_attn, kv_cache_type)

        # Process tokens one by one
        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION]
        ), torch.no_grad():
            for pos in range(seq_len):
                # Create input tensor for a single token
                x = torch.randn((self.batch_size, 1, self.dim), dtype=self.dtype)
                input_pos = torch.tensor([pos], dtype=torch.long)
                freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, 1)

                # Process with baseline attention
                baseline_out, _ = baseline_attn.forward(
                    x, freqs_cos, freqs_sin, input_pos=input_pos
                )

                # Process with ring buffer attention
                ring_out, _ = ring_attn.forward(
                    x, freqs_cos, freqs_sin, input_pos=input_pos
                )

                # Check that outputs are the same
                if kv_cache_type == KVCacheType.REGULAR:
                    self.assertTrue(
                        torch.allclose(baseline_out, ring_out, rtol=1e-7, atol=1e-7),
                        f"Outputs differ at position {pos}",
                    )
                else:
                    # For quantized kv cache we need bigger margin
                    self.assertTrue(
                        torch.allclose(baseline_out, ring_out, rtol=1e-6, atol=1e-6),
                        f"Outputs differ at position {pos}",
                    )

    def test_single_token_processing_quantized(self):
        """Test single token processing with QuantizedKVCache."""
        self._run_test_with_kv_cache_type(
            self.test_single_token_processing, KVCacheType.QUANTIZED
        )

    def test_single_token_processing_custom(self):
        """Test single token processing with CustomKVCache."""
        self._run_test_with_kv_cache_type(
            self.test_single_token_processing, KVCacheType.CUSTOM
        )

    def test_sliding_window_attention(
        self, kv_cache_type: KVCacheType = KVCacheType.REGULAR
    ):
        """Test that ring buffer with sliding window size produces the same output as baseline with sliding window mask."""
        self.sliding_window = 4
        self.max_context_len = 16

        seq_len = 10
        # Create baseline attention with full context length
        baseline_attn = self._create_baseline_attention(seq_len, kv_cache_type)

        # Create ring attention with sliding window size
        ring_attn = self._create_ring_attention(baseline_attn, kv_cache_type)

        # Process tokens one by one
        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION]
        ), torch.no_grad():
            for pos in range(seq_len):
                # Create input tensor for a single token
                x = torch.randn((self.batch_size, 1, self.dim), dtype=self.dtype)
                input_pos = torch.tensor([pos], dtype=torch.long)
                freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, 1)

                baseline_out, _ = baseline_attn.forward(
                    x, freqs_cos, freqs_sin, input_pos=input_pos
                )

                # Process with ring buffer attention
                ring_out, _ = ring_attn.forward(
                    x, freqs_cos, freqs_sin, input_pos=input_pos
                )

                # Check that outputs are the same
                self.assertTrue(
                    torch.allclose(baseline_out, ring_out, rtol=1e-7, atol=1e-7),
                    f"Outputs differ at position {pos}",
                )

    def test_sliding_window_attention_quantized(self):
        """Test sliding window attention with QuantizedKVCache."""
        self._run_test_with_kv_cache_type(
            self.test_sliding_window_attention, KVCacheType.QUANTIZED
        )

    def test_sliding_window_attention_custom(self):
        """Test sliding window attention with CustomKVCache."""
        self._run_test_with_kv_cache_type(
            self.test_sliding_window_attention, KVCacheType.CUSTOM
        )

    def test_ring_buffer_wrapping(
        self, kv_cache_type: KVCacheType = KVCacheType.REGULAR
    ):
        """Test that ring buffer correctly wraps around and maintains correct attention patterns."""
        self.sliding_window = 3
        self.max_context_len = 15

        # Create baseline attention with full context length
        baseline_attn = self._create_baseline_attention(
            self.max_context_len, kv_cache_type
        )

        # Create ring attention with sliding window size
        ring_attn = self._create_ring_attention(baseline_attn, kv_cache_type)

        # Process enough tokens to cause wrapping
        seq_len = 1
        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION]
        ), torch.no_grad():
            for pos in range(8):
                # Create input tensor for a single token
                x = torch.randn((self.batch_size, seq_len, self.dim), dtype=self.dtype)
                input_pos = torch.tensor([pos], dtype=torch.long)
                freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, seq_len)

                baseline_out, _ = baseline_attn.forward(
                    x, freqs_cos, freqs_sin, input_pos=input_pos
                )

                # Process with ring buffer attention
                ring_out, _ = ring_attn.forward(
                    x, freqs_cos, freqs_sin, input_pos=input_pos
                )
                self.assertTrue(
                    torch.allclose(baseline_out, ring_out, rtol=1e-7, atol=1e-7),
                    f"Outputs differ at position {pos}",
                )

        # After processing 8 tokens with window size 4, the ring buffer should have wrapped around
        # Check the cache positions to verify wrapping
        cache_positions = ring_attn.kv_cache.cache_positions_manager.cache_positions

        # The cache positions should contain the most recent 4 positions (4, 5, 6, 7)
        # mapped to the ring buffer indices
        expected_positions = torch.tensor([6, 7, 2, 3, 4, 5], dtype=torch.long)

        self.assertTrue(
            torch.all(cache_positions == expected_positions),
            f"Expected positions {expected_positions}, got {cache_positions}",
        )

    def test_ring_buffer_wrapping_quantized(self):
        """Test ring buffer wrapping with QuantizedKVCache."""
        self._run_test_with_kv_cache_type(
            self.test_ring_buffer_wrapping, KVCacheType.QUANTIZED
        )

    def test_ring_buffer_wrapping_custom(self):
        """Test ring buffer wrapping with CustomKVCache."""
        self._run_test_with_kv_cache_type(
            self.test_ring_buffer_wrapping, KVCacheType.CUSTOM
        )

    def test_large_context_with_sliding_window(
        self, kv_cache_type: KVCacheType = KVCacheType.REGULAR
    ):
        """Test with a large context length and compare baseline with sliding window to ring buffer."""
        # Use a larger context length and sliding window for this test
        self.max_context_len = 64
        self.sliding_window = 8

        token_lens = [8, 1, 3, 2, 1, 1, 1, 1, 7, 1, 5, 1, 1, 1, 4, 1, 1, 2, 1, 1]
        seq_len = sum(token_lens)
        # Create baseline attention with full context length
        baseline_attn = self._create_baseline_attention(seq_len, kv_cache_type)

        # Create ring attention with sliding window size
        ring_attn = self._create_ring_attention(baseline_attn, kv_cache_type)

        pos = 0
        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION]
        ), torch.no_grad():
            for token_len in token_lens:
                # Create input tensor for a single token
                x = torch.randn(
                    (self.batch_size, token_len, self.dim), dtype=self.dtype
                )
                input_pos = torch.tensor([pos], dtype=torch.long)
                freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, token_len)

                baseline_out, _ = baseline_attn.forward(
                    x, freqs_cos, freqs_sin, input_pos=input_pos
                )

                # Process with ring buffer attention
                ring_out, _ = ring_attn.forward(
                    x, freqs_cos, freqs_sin, input_pos=input_pos
                )

                # Check that outputs are the same
                self.assertTrue(
                    torch.allclose(baseline_out, ring_out, rtol=1e-7, atol=1e-7),
                    f"Outputs differ at position {pos} with max difference {(baseline_out - ring_out).abs().max()}",
                )
                pos += token_len

    def test_large_context_with_sliding_window_quantized(self):
        """Test large context with sliding window with QuantizedKVCache."""
        self._run_test_with_kv_cache_type(
            self.test_large_context_with_sliding_window, KVCacheType.QUANTIZED
        )

    def test_large_context_with_sliding_window_custom(self):
        """Test large context with sliding window with CustomKVCache."""
        self._run_test_with_kv_cache_type(
            self.test_large_context_with_sliding_window, KVCacheType.CUSTOM
        )
