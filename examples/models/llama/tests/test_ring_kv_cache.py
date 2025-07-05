# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.llama.attention import RingKVCache


class TestRingKVCache(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.max_batch_size = 2
        self.max_context_length = 8
        self.n_heads = 4
        self.head_dim = 16
        self.enable_dynamic_shape = True
        self.dtype = torch.float32

    def test_basic_update(self):
        """Test basic update functionality of RingKVCache."""
        cache = RingKVCache(
            self.max_batch_size,
            self.max_context_length,
            self.n_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            self.dtype,
        )

        # Create input tensors
        input_pos = torch.tensor([0], dtype=torch.long)
        seq_len = 3
        k_val = torch.ones(
            (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
            dtype=self.dtype,
        )
        v_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 2
        )

        # Update the cache
        k_out, v_out = cache.update(input_pos, k_val, v_val)

        # Check that the cache was updated correctly
        for i in range(seq_len):
            self.assertTrue(torch.all(k_out[:, :, i] == 1.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 2.0))

        # Check that the rest of the cache is still zeros
        for i in range(seq_len, self.max_context_length):
            self.assertTrue(torch.all(k_out[:, :, i] == 0.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 0.0))

        # Check that cache_positions was updated correctly
        expected_positions = torch.tensor(
            [0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            dtype=torch.long,
        )
        self.assertTrue(
            torch.all(
                cache.cache_positions_manager.cache_positions == expected_positions
            )
        )

    def test_ring_buffer_wrapping(self):
        """Test that the ring buffer wraps around correctly."""
        cache = RingKVCache(
            self.max_batch_size,
            self.max_context_length,
            self.n_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            self.dtype,
        )

        # Create input tensors for first update
        input_pos = torch.tensor([14], dtype=torch.long)
        seq_len = 4  # This will wrap around from position 14 to positions 14, 15, 0, 1
        k_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 3
        )
        v_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 4
        )

        # Update the cache
        k_out, v_out = cache.update(input_pos, k_val, v_val)

        # Check that the cache was updated correctly with wrapping
        # Positions 14, 15 should be updated
        for i in range(14, 16):
            self.assertTrue(torch.all(k_out[:, :, i] == 3.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 4.0))

        # Positions 0, 1 should also be updated due to wrapping
        for i in range(0, 2):
            self.assertTrue(torch.all(k_out[:, :, i] == 3.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 4.0))

        # The rest should still be zeros
        for i in range(2, 14):
            self.assertTrue(torch.all(k_out[:, :, i] == 0.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 0.0))

        # Check that cache_positions was updated correctly
        # Note that positions 2-13 are 0 instead of -1 because in actual ring
        # updates those positions would have been updated.
        # But CachePositionsManager thinks they are updated because start_pos > (2-13)
        # As a result it does not fill them with -1 and instead uses original values
        # which is 0, the value cache_position buffer is initialized with.
        expected_positions = torch.tensor(
            [16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 15], dtype=torch.long
        )
        self.assertTrue(
            torch.all(
                cache.cache_positions_manager.cache_positions == expected_positions
            )
        )

    def test_multiple_updates(self):
        """Test multiple updates to the cache."""
        cache = RingKVCache(
            self.max_batch_size,
            self.max_context_length,
            self.n_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            self.dtype,
        )

        # First update
        input_pos1 = torch.tensor([0], dtype=torch.long)
        seq_len1 = 2
        k_val1 = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len1, self.head_dim),
                dtype=self.dtype,
            )
            * 5
        )
        v_val1 = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len1, self.head_dim),
                dtype=self.dtype,
            )
            * 6
        )

        _, _ = cache.update(input_pos1, k_val1, v_val1)

        # Second update
        input_pos2 = torch.tensor([2], dtype=torch.long)
        seq_len2 = 3
        k_val2 = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len2, self.head_dim),
                dtype=self.dtype,
            )
            * 7
        )
        v_val2 = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len2, self.head_dim),
                dtype=self.dtype,
            )
            * 8
        )

        k_out2, v_out2 = cache.update(input_pos2, k_val2, v_val2)

        # Check that the cache was updated correctly after both updates
        # First update (positions 0, 1)
        for i in range(0, 2):
            self.assertTrue(torch.all(k_out2[:, :, i] == 5.0))
            self.assertTrue(torch.all(v_out2[:, :, i] == 6.0))

        # Second update (positions 2, 3, 4)
        for i in range(2, 5):
            self.assertTrue(torch.all(k_out2[:, :, i] == 7.0))
            self.assertTrue(torch.all(v_out2[:, :, i] == 8.0))

        # The rest should still be zeros
        for i in range(5, 8):
            self.assertTrue(torch.all(k_out2[:, :, i] == 0.0))
            self.assertTrue(torch.all(v_out2[:, :, i] == 0.0))

        # Check that cache_positions was updated correctly
        expected_positions = torch.tensor(
            [0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            dtype=torch.long,
        )
        self.assertTrue(
            torch.all(
                cache.cache_positions_manager.cache_positions == expected_positions
            )
        )

        # Third update with wrapping
        input_pos3 = torch.tensor([14], dtype=torch.long)
        seq_len3 = 4
        k_val3 = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len3, self.head_dim),
                dtype=self.dtype,
            )
            * 9
        )
        v_val3 = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len3, self.head_dim),
                dtype=self.dtype,
            )
            * 10
        )

        k_out3, v_out3 = cache.update(input_pos3, k_val3, v_val3)

        # Check final state after third update with wrapping
        # Positions 0, 1 should now have values from the third update (due to wrapping)
        for i in range(0, 2):
            self.assertTrue(torch.all(k_out3[:, :, i] == 9.0))
            self.assertTrue(torch.all(v_out3[:, :, i] == 10.0))

        # Positions 2, 3, 4 should still have values from the second update
        for i in range(2, 5):
            self.assertTrue(torch.all(k_out3[:, :, i] == 7.0))
            self.assertTrue(torch.all(v_out3[:, :, i] == 8.0))

        # Positions 5-13 should still be zero
        for i in range(5, 14):
            self.assertTrue(torch.all(k_out3[:, :, i] == 0.0))
            self.assertTrue(torch.all(v_out3[:, :, i] == 0.0))

        # Positions 14, 15 should have values from the third update
        for i in range(14, 16):
            self.assertTrue(torch.all(k_out3[:, :, i] == 9.0))
            self.assertTrue(torch.all(v_out3[:, :, i] == 10.0))

        # Check that cache_positions was updated correctly
        expected_positions = torch.tensor(
            [16, 17, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 15],
            dtype=torch.long,
        )
        self.assertTrue(
            torch.all(
                cache.cache_positions_manager.cache_positions == expected_positions
            )
        )

    def test_edge_case_input_pos_zero(self):
        """Test the edge case where input_pos is 0."""
        cache = RingKVCache(
            self.max_batch_size,
            self.max_context_length,
            self.n_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            self.dtype,
        )

        # Create input tensors
        input_pos = torch.tensor([0], dtype=torch.long)
        seq_len = 1
        k_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 11
        )
        v_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 12
        )

        # Update the cache
        k_out, v_out = cache.update(input_pos, k_val, v_val)

        # Check that position 0 was updated
        self.assertTrue(torch.all(k_out[:, :, 0] == 11.0))
        self.assertTrue(torch.all(v_out[:, :, 0] == 12.0))

        # Check that the rest of the cache is still zeros
        for i in range(1, self.max_context_length):
            self.assertTrue(torch.all(k_out[:, :, i] == 0.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 0.0))

        # Check that cache_positions was updated correctly
        expected_positions = torch.tensor(
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            dtype=torch.long,
        )
        self.assertTrue(
            torch.all(
                cache.cache_positions_manager.cache_positions == expected_positions
            )
        )

    def test_edge_case_exceeding_context_length(self):
        """Test the edge case where input_pos + seq_len > max_context_length."""
        cache = RingKVCache(
            self.max_batch_size,
            self.max_context_length,
            self.n_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            self.dtype,
        )

        # Create input tensors
        input_pos = torch.tensor([13], dtype=torch.long)
        seq_len = (
            5  # This will wrap around from position 13 to positions 13, 14, 15, 0, 1
        )
        k_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 13
        )
        v_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 14
        )

        # Update the cache
        k_out, v_out = cache.update(input_pos, k_val, v_val)

        # Check that positions 13, 14, 15 were updated
        for i in range(13, 16):
            self.assertTrue(torch.all(k_out[:, :, i] == 13.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 14.0))

        # Check that positions 0, 1 were also updated due to wrapping
        for i in range(0, 2):
            self.assertTrue(torch.all(k_out[:, :, i] == 13.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 14.0))

        # Check that positions 2-12 are still zeros
        for i in range(2, 13):
            self.assertTrue(torch.all(k_out[:, :, i] == 0.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 0.0))

        # Check that cache_positions was updated correctly
        # Note that positions 2-12 are 0 instead of -1 because in actual ring
        # updates those positions would have been updated.
        # But CachePositionsManager thinks they are updated because start_pos > (2-12)
        # As a result it does not fill them with -1 and instead uses original values
        # which is 0, the value cache_position buffer is initialized with.
        expected_positions = torch.tensor(
            [16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 15], dtype=torch.long
        )
        self.assertTrue(
            torch.all(
                cache.cache_positions_manager.cache_positions == expected_positions
            )
        )

    def test_original_indices_tracking(self):
        """Test that the original indices are tracked correctly in cache_positions."""
        cache = RingKVCache(
            self.max_batch_size,
            self.max_context_length,
            self.n_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            self.dtype,
        )

        # First update at position 10 (will be mapped to position 10 in the ring buffer)
        input_pos = torch.tensor([10], dtype=torch.long)
        seq_len = 4
        k_val = torch.ones(
            (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
            dtype=self.dtype,
        )
        v_val = torch.ones(
            (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
            dtype=self.dtype,
        )

        # Update the cache
        cache.update(input_pos, k_val, v_val)

        # Check that cache_positions correctly tracks the original indices
        # For input_pos=10 and seq_len=4, the original indices should be 10, 11, 12, 13
        # These map to positions 10, 11, 12, 13 in the ring buffer (since max_context_length=8 but buffer size is 16)
        # Note that positions 0-9 are 0 because in actual ring
        # updates those positions would have been updated for start_pos = 0.
        # So CachePositionsManager thinks they are updated because start_pos > (0-9)
        # As a result it does not fill them with -1 and instead uses original values
        # which is 0, the value cache_position buffer is initialized with.
        expected_positions = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, -1, -1], dtype=torch.long
        )
        self.assertTrue(
            torch.all(
                cache.cache_positions_manager.cache_positions == expected_positions
            )
        )

        # Second update at position 14 (will be mapped to position 14 in the ring buffer)
        input_pos = torch.tensor([14], dtype=torch.long)
        seq_len = 3
        k_val = torch.ones(
            (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
            dtype=self.dtype,
        )
        v_val = torch.ones(
            (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
            dtype=self.dtype,
        )

        # Update the cache
        cache.update(input_pos, k_val, v_val)

        # Check that cache_positions correctly tracks the original indices
        # For input_pos=14 and seq_len=3, the original indices should be 14, 15, 16
        # These map to positions 14, 15, 0 in the ring buffer
        expected_positions = torch.tensor(
            [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15], dtype=torch.long
        )
        self.assertTrue(
            torch.all(
                cache.cache_positions_manager.cache_positions == expected_positions
            )
        )

    def test_non_dynamic_shape(self):
        """Test RingKVCache with enable_dynamic_shape=False."""
        cache = RingKVCache(
            self.max_batch_size,
            self.max_context_length,
            self.n_heads,
            self.head_dim,
            enable_dynamic_shape=False,
            dtype=self.dtype,
        )

        # Create input tensors
        input_pos = torch.tensor([0], dtype=torch.long)
        seq_len = 3
        k_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 15
        )
        v_val = (
            torch.ones(
                (self.max_batch_size, self.n_heads, seq_len, self.head_dim),
                dtype=self.dtype,
            )
            * 16
        )

        # Update the cache
        k_out, v_out = cache.update(input_pos, k_val, v_val)

        # Check that the cache was updated correctly
        for i in range(seq_len):
            self.assertTrue(torch.all(k_out[:, :, i] == 15.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 16.0))

        # Check that the rest of the cache is still zeros
        for i in range(seq_len, self.max_context_length):
            self.assertTrue(torch.all(k_out[:, :, i] == 0.0))
            self.assertTrue(torch.all(v_out[:, :, i] == 0.0))
