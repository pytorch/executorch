# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the ring-buffer based attention sink implementation.

This tests the torch.export-compatible implementation that uses a ring buffer
for the sliding window rather than explicit token eviction.

Usage:
    # Run with pytest
    python -m pytest examples/models/llama/source_transformation/test_attention_sink_ring_buffer.py -v
    
    # Or run directly
    python examples/models/llama/source_transformation/test_attention_sink_ring_buffer.py
"""

import unittest

import torch
from executorch.examples.models.llama.model_args import ModelArgs

from executorch.examples.models.llama.source_transformation.attention_sink import (
    CachePositionsManagerWithSink,
    KVCacheWithAttentionSink,
    RopeWithAttentionSink,
    _create_causal_mask_for_attention_sink,
)


class CachePositionsManagerWithSinkTest(unittest.TestCase):
    """Test the cache positions manager for ring buffer indexing."""

    def setUp(self):
        self.cache_size = 32  # Total cache size (e.g., sink_size + window_size * 2)
        # Default: no sink (simple ring buffer)
        self.manager = CachePositionsManagerWithSink(self.cache_size, sink_size=0)

    def test_initial_positions_are_minus_one(self):
        """Cache positions should start as -1 (unwritten)."""
        expected = torch.full((self.cache_size,), -1, dtype=torch.long)
        torch.testing.assert_close(self.manager.cache_positions, expected)

    def test_simple_update(self):
        """Test simple sequential update without wraparound."""
        input_pos = torch.tensor([0], dtype=torch.long)
        seq_len = 5
        indices = self.manager.calculate_positions_and_update_indices(input_pos, seq_len)

        # Should return indices 0, 1, 2, 3, 4
        expected_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        torch.testing.assert_close(indices, expected_indices)

        # Cache positions at those indices should be the original positions
        for i in range(5):
            self.assertEqual(self.manager.cache_positions[i].item(), i)

    def test_wraparound_no_sink(self):
        """Test ring buffer wraparound with sink_size=0."""
        # Fill cache to position 30
        input_pos = torch.tensor([0], dtype=torch.long)
        self.manager.calculate_positions_and_update_indices(input_pos, 30)

        # Add 5 more tokens at position 30 - should wrap around
        input_pos = torch.tensor([30], dtype=torch.long)
        indices = self.manager.calculate_positions_and_update_indices(input_pos, 5)

        # Indices should wrap: 30 % 32 = 30, 31, 0, 1, 2
        expected_indices = torch.tensor([30, 31, 0, 1, 2], dtype=torch.long)
        torch.testing.assert_close(indices, expected_indices)

    def test_wraparound_with_sink(self):
        """Test ring buffer wraparound with sink_size > 0."""
        sink_size = 4
        cache_size = 32
        manager = CachePositionsManagerWithSink(cache_size, sink_size)
        
        # Fill cache to position 30
        input_pos = torch.tensor([0], dtype=torch.long)
        manager.calculate_positions_and_update_indices(input_pos, 30)

        # Add 5 more tokens at position 30
        input_pos = torch.tensor([30], dtype=torch.long)
        indices = manager.calculate_positions_and_update_indices(input_pos, 5)

        # Ring size = 32 - 4 = 28
        # pos 30 -> idx = 4 + (30-4)%28 = 4 + 26 = 30
        # pos 31 -> idx = 4 + (31-4)%28 = 4 + 27 = 31
        # pos 32 -> idx = 4 + (32-4)%28 = 4 + 0 = 4 (WRAPS TO SINK_SIZE=4, not 0!)
        # pos 33 -> idx = 4 + (33-4)%28 = 4 + 1 = 5
        # pos 34 -> idx = 4 + (34-4)%28 = 4 + 2 = 6
        expected_indices = torch.tensor([30, 31, 4, 5, 6], dtype=torch.long)
        torch.testing.assert_close(indices, expected_indices)

    def test_cache_positions_track_original_positions_no_sink(self):
        """Cache positions should track which original position is at each index (no sink)."""
        # Fill with positions 0-31
        input_pos = torch.tensor([0], dtype=torch.long)
        self.manager.calculate_positions_and_update_indices(input_pos, 32)

        # Now add position 32 which wraps to index 0
        input_pos = torch.tensor([32], dtype=torch.long)
        self.manager.calculate_positions_and_update_indices(input_pos, 1)

        # Index 0 should now contain original position 32
        self.assertEqual(self.manager.cache_positions[0].item(), 32)

    def test_cache_positions_track_original_positions_with_sink(self):
        """Cache positions should track positions, and sink tokens are never overwritten."""
        sink_size = 4
        cache_size = 32
        manager = CachePositionsManagerWithSink(cache_size, sink_size)
        
        # Fill with positions 0-31
        input_pos = torch.tensor([0], dtype=torch.long)
        manager.calculate_positions_and_update_indices(input_pos, 32)
        
        # Indices 0-3 should have pos 0-3 (Sink tokens)
        for i in range(4):
            self.assertEqual(manager.cache_positions[i].item(), i)
        
        # Now add position 32.
        # (32-4)%28 = 0. So index = 4 + 0 = 4.
        input_pos = torch.tensor([32], dtype=torch.long)
        manager.calculate_positions_and_update_indices(input_pos, 1)
        
        # Index 4 should now contain original position 32
        self.assertEqual(manager.cache_positions[4].item(), 32)
        
        # Index 0-3 (sink) should STILL contain positions 0-3 (unchanged)
        for i in range(4):
            self.assertEqual(manager.cache_positions[i].item(), i)


class CausalMaskTest(unittest.TestCase):
    """Test the causal mask generation for attention sink."""

    def test_mask_allows_sink_tokens(self):
        """Sink tokens should always be visible (mask = 0)."""
        cache_size = 32
        sink_size = 4
        window_size = 14  # cache_size = sink_size + window_size * 2
        
        # Create cache positions where positions 0-3 are sink tokens
        cache_positions = torch.arange(cache_size, dtype=torch.long)
        
        start_pos = 20  # Query at position 20
        seq_len = 1
        
        mask = _create_causal_mask_for_attention_sink(
            cache_positions, window_size, sink_size, start_pos, seq_len
        )
        
        # Sink tokens (indices 0-3, original positions 0-3) should have mask = 0
        for i in range(sink_size):
            self.assertEqual(mask[0, i].item(), 0.0, f"Sink token at index {i} should be visible")

    def test_mask_blocks_future_tokens(self):
        """Future tokens should be masked (-inf)."""
        cache_size = 32
        sink_size = 4
        window_size = 14
        
        cache_positions = torch.arange(cache_size, dtype=torch.long)
        
        start_pos = 10
        seq_len = 1
        
        mask = _create_causal_mask_for_attention_sink(
            cache_positions, window_size, sink_size, start_pos, seq_len
        )
        
        # Future tokens (positions > 10) should have mask = -inf
        for i in range(11, cache_size):
            self.assertEqual(mask[0, i].item(), float('-inf'), f"Future token at position {i} should be masked")

    def test_mask_respects_window(self):
        """Tokens outside the window should be masked."""
        cache_size = 32
        sink_size = 4
        window_size = 5  # Only allow 5 recent tokens
        
        cache_positions = torch.arange(cache_size, dtype=torch.long)
        
        start_pos = 20
        seq_len = 1
        
        mask = _create_causal_mask_for_attention_sink(
            cache_positions, window_size, sink_size, start_pos, seq_len
        )
        
        # Positions 16-20 should be visible (within window of 5)
        for pos in range(16, 21):
            self.assertEqual(mask[0, pos].item(), 0.0, f"Position {pos} should be visible (in window)")
        
        # Position 15 should be masked (outside window, not a sink)
        self.assertEqual(mask[0, 15].item(), float('-inf'), f"Position 15 should be masked (outside window)")


class KVCacheWithAttentionSinkTest(unittest.TestCase):
    """Test the KV cache with attention sink."""

    def setUp(self):
        torch.manual_seed(42)
        self.window_size = 14
        self.sink_size = 4
        self.n_heads = 8
        self.head_dim = 64
        self.max_batch_size = 1
        
        # Create model args with enough context for RoPE
        self.params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_context_len=1024,  # Large enough for RoPE
            n_heads=self.n_heads,
            n_kv_heads=self.n_heads,
            dim=self.n_heads * self.head_dim,
        )
        
        self.rope = RopeWithAttentionSink(
            params=self.params,
            window_size=self.window_size,
            sink_size=self.sink_size,
            eviction_batch_size=1,
        )
        
        self.kv_cache = KVCacheWithAttentionSink(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            enable_dynamic_shape=True,
            rope=self.rope,
            max_batch_size=self.max_batch_size,
            window_size=self.window_size,
            sink_size=self.sink_size,
            eviction_batch_size=1,
            dtype=torch.float32,
        )

    def test_cache_size(self):
        """Cache should be sink_size + window_size * 2."""
        expected_size = self.sink_size + self.window_size * 2  # 4 + 28 = 32
        self.assertEqual(self.kv_cache.k_cache.size(2), expected_size)
        self.assertEqual(self.kv_cache.v_cache.size(2), expected_size)

    def test_is_ring_buffer(self):
        """Cache should be marked as ring buffer."""
        self.assertTrue(self.kv_cache.is_ring_buffer)

    def test_update_stores_kv(self):
        """Update should store key-value pairs."""
        k = torch.randn(1, self.n_heads, 5, self.head_dim)
        v = torch.randn(1, self.n_heads, 5, self.head_dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        
        k_out, v_out = self.kv_cache.update(input_pos, k, v)
        
        # First 5 positions should contain our values
        torch.testing.assert_close(k_out[:, :, :5, :], k)
        torch.testing.assert_close(v_out[:, :, :5, :], v)

    def test_evict_tokens_returns_zero(self):
        """Ring buffer implementation doesn't shift, so evict returns 0."""
        input_pos = torch.tensor([100], dtype=torch.long)
        shift = self.kv_cache.evict_tokens(input_pos, 10)
        self.assertEqual(shift, 0)

    def test_extended_generation(self):
        """Test that cache works for positions beyond cache size."""
        cache_size = self.kv_cache.k_cache.size(2)
        
        # Fill cache with initial tokens
        for pos in range(cache_size + 50):
            k = torch.randn(1, self.n_heads, 1, self.head_dim)
            v = torch.randn(1, self.n_heads, 1, self.head_dim)
            input_pos = torch.tensor([pos], dtype=torch.long)
            
            k_out, v_out = self.kv_cache.update(input_pos, k, v)
            
            # Should not raise any errors
            self.assertEqual(k_out.shape, self.kv_cache.k_cache.shape)
            self.assertEqual(v_out.shape, self.kv_cache.v_cache.shape)


class RopeWithAttentionSinkTest(unittest.TestCase):
    """Test RoPE for attention sink."""

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_context_len=1024,
            n_heads=8,
            dim=512,
        )
        
        self.rope = RopeWithAttentionSink(
            params=self.params,
            window_size=100,
            sink_size=4,
            eviction_batch_size=1,
        )

    def test_get_freqs_uses_original_position(self):
        """RoPE frequencies should use the original position."""
        input_pos = torch.tensor([50], dtype=torch.long)
        seq_len = 5
        
        freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, seq_len)
        
        # Should get frequencies for positions 50-54
        expected_cos = self.rope.freqs_cos[50:55]
        expected_sin = self.rope.freqs_sin[50:55]
        
        torch.testing.assert_close(freqs_cos, expected_cos)
        torch.testing.assert_close(freqs_sin, expected_sin)

    def test_rerotate_k(self):
        """Test re-rotation of k from one position to another."""
        batch_size = 1
        seq_len = 8
        n_heads = self.params.n_heads
        head_dim = self.params.dim // n_heads
        
        k = torch.randn(batch_size, seq_len, n_heads, head_dim)
        q = torch.randn(batch_size, seq_len, n_heads, head_dim)
        
        # Rotate k at position 100
        original_pos = 100
        freqs_cos, freqs_sin = self.rope.get_freqs(
            torch.tensor([original_pos], dtype=torch.long), seq_len
        )
        _, rotated_k = self.rope.forward(q, k, freqs_cos, freqs_sin)
        
        # Re-rotate to position 50
        new_pos = 50
        rerotated_k = self.rope.rerotate_k(rotated_k, original_pos, new_pos)
        
        # This should be equivalent to directly rotating k at position 50
        freqs_cos_new, freqs_sin_new = self.rope.get_freqs(
            torch.tensor([new_pos], dtype=torch.long), seq_len
        )
        _, expected_k = self.rope.forward(q, k, freqs_cos_new, freqs_sin_new)
        
        torch.testing.assert_close(rerotated_k, expected_k, rtol=1e-4, atol=1e-4)


class CausalMaskWithWraparoundTest(unittest.TestCase):
    """Test causal mask with ring buffer wraparound."""

    def test_mask_after_wraparound(self):
        """Test mask after cache has wrapped around."""
        cache_size = 16
        sink_size = 4
        window_size = 6  # cache_size = sink_size + window_size * 2
        
        # Simulate cache after generating beyond cache_size:
        # The ring buffer wraps, so indices 0-15 contain positions that wrap
        # At position 50, with cache_size=16, the cache contains:
        # positions 50-15=35 to 49 at various indices
        cache_positions = torch.zeros(cache_size, dtype=torch.long)
        # Fill with positions that would exist after generating 50 tokens
        # idx = pos % cache_size, so:
        # pos 34-49 occupy indices 2-15 and 0-1
        for pos in range(34, 50):
            idx = pos % cache_size
            cache_positions[idx] = pos
        
        start_pos = 49  # Query at position 49
        seq_len = 1
        
        mask = _create_causal_mask_for_attention_sink(
            cache_positions, window_size, sink_size, start_pos, seq_len
        )
        
        # Positions within window (49-6+1=44 to 49) should be visible
        visible_count = 0
        for i in range(cache_size):
            pos = cache_positions[i].item()
            if pos >= 44 and pos <= 49:  # In window
                self.assertEqual(mask[0, i].item(), 0.0, 
                               f"Position {pos} at idx {i} should be visible (in window)")
                visible_count += 1
        
        # Should have some visible tokens
        self.assertGreater(visible_count, 0, "Should have visible tokens in window")

    def test_mask_with_sink_size_zero(self):
        """Test pure sliding window (sink_size=0)."""
        cache_size = 16
        sink_size = 0
        window_size = 8
        
        cache_positions = torch.arange(cache_size, dtype=torch.long)
        start_pos = 10
        seq_len = 1
        
        mask = _create_causal_mask_for_attention_sink(
            cache_positions, window_size, sink_size, start_pos, seq_len
        )
        
        # Positions 3-10 should be visible (within window of 8)
        for pos in range(3, 11):
            self.assertEqual(mask[0, pos].item(), 0.0, f"Position {pos} should be visible")
        
        # Positions 0-2 should be masked (outside window)
        for pos in range(0, 3):
            self.assertEqual(mask[0, pos].item(), float('-inf'), 
                           f"Position {pos} should be masked (outside window)")


class PrefillTest(unittest.TestCase):
    """Test prefill scenarios."""

    def setUp(self):
        torch.manual_seed(42)
        self.window_size = 14
        self.sink_size = 4
        self.n_heads = 8
        self.head_dim = 64
        
        self.params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_context_len=1024,
            n_heads=self.n_heads,
            n_kv_heads=self.n_heads,
            dim=self.n_heads * self.head_dim,
        )
        
        self.rope = RopeWithAttentionSink(
            params=self.params,
            window_size=self.window_size,
            sink_size=self.sink_size,
            eviction_batch_size=1,
        )
        
        self.kv_cache = KVCacheWithAttentionSink(
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            enable_dynamic_shape=True,
            rope=self.rope,
            max_batch_size=1,
            window_size=self.window_size,
            sink_size=self.sink_size,
            eviction_batch_size=1,
            dtype=torch.float32,
        )

    def test_prefill_entire_cache(self):
        """Test prefill that fills entire cache."""
        cache_size = self.kv_cache.k_cache.size(2)
        
        k = torch.randn(1, self.n_heads, cache_size, self.head_dim)
        v = torch.randn(1, self.n_heads, cache_size, self.head_dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        
        k_out, v_out = self.kv_cache.update(input_pos, k, v)
        
        # All positions should be filled
        torch.testing.assert_close(k_out, k)
        torch.testing.assert_close(v_out, v)

    def test_prefill_larger_than_cache_raises_error(self):
        """Test that prefill larger than cache size raises an assertion error."""
        cache_size = self.kv_cache.k_cache.size(2)
        seq_len = cache_size + 10
        
        k = torch.randn(1, self.n_heads, seq_len, self.head_dim)
        v = torch.randn(1, self.n_heads, seq_len, self.head_dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        
        # This should raise an assertion error since seq_len > cache_size
        with self.assertRaises(AssertionError):
            self.kv_cache.update(input_pos, k, v)

    def test_prefill_followed_by_decode(self):
        """Test prefill followed by decode steps."""
        cache_size = self.kv_cache.k_cache.size(2)
        
        # Prefill with 20 tokens
        k_prefill = torch.randn(1, self.n_heads, 20, self.head_dim)
        v_prefill = torch.randn(1, self.n_heads, 20, self.head_dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        self.kv_cache.update(input_pos, k_prefill, v_prefill)
        
        # Decode 5 more tokens
        for i in range(5):
            k_decode = torch.randn(1, self.n_heads, 1, self.head_dim)
            v_decode = torch.randn(1, self.n_heads, 1, self.head_dim)
            input_pos = torch.tensor([20 + i], dtype=torch.long)
            k_out, v_out = self.kv_cache.update(input_pos, k_decode, v_decode)
            
            # Verify cache positions are updated
            expected_pos = 20 + i
            cache_idx = expected_pos % cache_size
            self.assertEqual(
                self.kv_cache.cache_positions_manager.cache_positions[cache_idx].item(),
                expected_pos
            )


class EnableAttentionSinkTest(unittest.TestCase):
    """Test the enable_attention_sink transformation."""

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_context_len=1024,
            n_heads=8,
            n_kv_heads=8,
            dim=512,
            n_layers=2,
            vocab_size=100,
        )

    def test_enable_attention_sink_transforms_model(self):
        """Test that enable_attention_sink properly transforms the model."""
        from executorch.examples.models.llama.llama_transformer import construct_transformer
        from executorch.examples.models.llama.source_transformation.attention_sink import (
            enable_attention_sink,
        )
        
        # Create a simple transformer
        with torch.device("meta"):
            model = construct_transformer(self.params)
        model.to_empty(device="cpu")
        
        # Apply attention sink transformation
        model = enable_attention_sink(
            module=model,
            params=self.params,
            sink_size=4,
            window_size=100,
            eviction_batch_size=1,
        )
        
        # Check that KV caches are replaced
        for layer in model.layers:
            kv_cache = layer.attention.kv_cache
            self.assertIsInstance(kv_cache, KVCacheWithAttentionSink)
            self.assertEqual(kv_cache.sink_size, 4)
            self.assertEqual(kv_cache.window_size, 100)
            self.assertTrue(kv_cache.is_ring_buffer)

    def test_enable_attention_sink_replaces_rope(self):
        """Test that RoPE is replaced with RopeWithAttentionSink."""
        from executorch.examples.models.llama.llama_transformer import construct_transformer
        from executorch.examples.models.llama.source_transformation.attention_sink import (
            enable_attention_sink,
        )
        
        with torch.device("meta"):
            model = construct_transformer(self.params)
        model.to_empty(device="cpu")
        
        model = enable_attention_sink(
            module=model,
            params=self.params,
            sink_size=4,
            window_size=100,
            eviction_batch_size=1,
        )
        
        # Check that rope is replaced
        for layer in model.layers:
            rope = layer.attention.rope
            self.assertIsInstance(rope, RopeWithAttentionSink)


class IntegrationTest(unittest.TestCase):
    """Integration tests for end-to-end scenarios."""

    def setUp(self):
        torch.manual_seed(42)

    def test_cache_positions_consistency(self):
        """Test that cache positions remain consistent during generation."""
        cache_size = 32
        sink_size = 4
        window_size = 14
        n_heads = 8
        head_dim = 64
        
        params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_context_len=1024,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            dim=n_heads * head_dim,
        )
        
        rope = RopeWithAttentionSink(
            params=params,
            window_size=window_size,
            sink_size=sink_size,
            eviction_batch_size=1,
        )
        
        kv_cache = KVCacheWithAttentionSink(
            n_heads=n_heads,
            head_dim=head_dim,
            enable_dynamic_shape=True,
            rope=rope,
            max_batch_size=1,
            window_size=window_size,
            sink_size=sink_size,
            eviction_batch_size=1,
            dtype=torch.float32,
        )
        
        # Generate 100 tokens
        for pos in range(100):
            k = torch.randn(1, n_heads, 1, head_dim)
            v = torch.randn(1, n_heads, 1, head_dim)
            input_pos = torch.tensor([pos], dtype=torch.long)
            
            kv_cache.update(input_pos, k, v)
            
            # Create mask and verify it's valid
            mask = kv_cache.create_causal_mask_for_ring_buffer(pos, 1)
            
            # Mask should not be all -inf (would mean no tokens to attend to)
            non_inf_count = (mask != float('-inf')).sum().item()
            self.assertGreater(non_inf_count, 0, f"At pos {pos}, mask should have visible tokens")
            
            # For positions >= sink_size, sinks should always be visible
            if pos >= sink_size:
                for i in range(sink_size):
                    cache_pos = kv_cache.cache_positions_manager.cache_positions[i].item()
                    if cache_pos < sink_size:  # This is actually a sink token
                        self.assertEqual(mask[0, i].item(), 0.0,
                                       f"Sink at idx {i} should be visible at pos {pos}")


if __name__ == '__main__':
    unittest.main()
