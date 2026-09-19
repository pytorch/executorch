# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import multiprocessing
import unittest

import torch

from executorch.extension.llm.custom_ops import custom_ops  # noqa


def run_in_subprocess(target):
    """
    Decorator to run the target function in a separate subprocess
    so as to allow cpp code to throw runtime::abort
    """

    def wrapper(*args, **kwargs):
        p = multiprocessing.Process(target=target, args=args, kwargs=kwargs)
        p.start()
        p.join()
        if p.exitcode != 0:
            raise Exception(f"Subprocess failed with exit code {p.exitcode}")

    return wrapper


class UpdateQuantizedKVCacheTest(unittest.TestCase):

    def _reset(self):
        self.quantized_k_cache = torch.zeros(
            (self.batch_size, self.seq_len, self.num_heads, self.head_dim),
            dtype=torch.int8,
        )
        self.quantized_v_cache = torch.zeros(
            (self.batch_size, self.seq_len, self.num_heads, self.head_dim),
            dtype=torch.int8,
        )
        self.k_scales_cache = torch.zeros(
            (self.batch_size, self.seq_len, self.num_heads, 1), dtype=torch.float64
        )
        self.v_scales_cache = torch.zeros(
            (self.batch_size, self.seq_len, self.num_heads, 1), dtype=torch.float64
        )
        self.k_zero_points_cache = torch.zeros(
            (self.batch_size, self.seq_len, self.num_heads, 1), dtype=torch.int64
        )
        self.v_zero_points_cache = torch.zeros(
            (self.batch_size, self.seq_len, self.num_heads, 1), dtype=torch.int64
        )

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 1
        self.seq_len = 10
        self.num_heads = 8
        self.head_dim = 4
        self._reset()

    def _update_k(self, start_pos, value, scales, zero_points):
        seq_len = value.size(1)
        self.quantized_k_cache[:, start_pos : start_pos + seq_len, :, :] = value
        self.k_scales_cache[:, start_pos : start_pos + seq_len, :, :] = scales
        self.k_zero_points_cache[:, start_pos : start_pos + seq_len, :, :] = zero_points

    def _update_v(self, start_pos, value, scales, zero_points):
        seq_len = value.size(1)
        self.quantized_v_cache[:, start_pos : start_pos + seq_len, :, :] = value
        self.v_scales_cache[:, start_pos : start_pos + seq_len, :, :] = scales
        self.v_zero_points_cache[:, start_pos : start_pos + seq_len, :, :] = zero_points

    def _update_and_validate(
        self, k, v, k_scales, v_scales, k_zero_points, v_zero_points, start_pos
    ):
        k_cache = self.quantized_k_cache.clone()
        v_cache = self.quantized_v_cache.clone()
        k_scales_cache = self.k_scales_cache.clone()
        v_scales_cache = self.v_scales_cache.clone()
        k_zero_points_cache = self.k_zero_points_cache.clone()
        v_zero_points_cache = self.v_zero_points_cache.clone()
        self._update_k(start_pos, k, k_scales, k_zero_points)
        self._update_v(start_pos, v, v_scales, v_zero_points)

        torch.ops.llama.update_cache(k, k_cache, start_pos)
        torch.ops.llama.update_cache(k_scales, k_scales_cache, start_pos)
        torch.ops.llama.update_cache(k_zero_points, k_zero_points_cache, start_pos)

        torch.ops.llama.update_cache(v, v_cache, start_pos)
        torch.ops.llama.update_cache(v_scales, v_scales_cache, start_pos)
        torch.ops.llama.update_cache(v_zero_points, v_zero_points_cache, start_pos)

        self.assertTrue(torch.allclose(k_cache, self.quantized_k_cache))
        self.assertTrue(torch.allclose(v_cache, self.quantized_v_cache))
        self.assertTrue(torch.allclose(k_scales_cache, self.k_scales_cache))
        self.assertTrue(torch.allclose(v_scales_cache, self.v_scales_cache))
        self.assertTrue(torch.allclose(k_zero_points_cache, self.k_zero_points_cache))
        self.assertTrue(torch.allclose(v_zero_points_cache, self.v_zero_points_cache))

    def _update_with_indices_and_validate(
        self, k, k_scales, k_zero_points, start_pos, indices
    ):
        k_cache = self.quantized_k_cache.clone()
        k_scales_cache = self.k_scales_cache.clone()
        k_zero_points_cache = self.k_zero_points_cache.clone()

        # Update using Python indexing for reference
        for batch_idx in range(self.batch_size):
            for seq_idx in range(indices.size(1)):
                idx = indices[batch_idx, seq_idx].item()
                if idx >= 0 and idx < self.seq_len:
                    self.quantized_k_cache[batch_idx, idx] = k[batch_idx, seq_idx]
                    self.k_scales_cache[batch_idx, idx] = k_scales[batch_idx, seq_idx]
                    self.k_zero_points_cache[batch_idx, idx] = k_zero_points[
                        batch_idx, seq_idx
                    ]

        # Update using custom op
        torch.ops.llama.update_cache_with_indices(k, k_cache, start_pos, indices)
        torch.ops.llama.update_cache_with_indices(
            k_scales, k_scales_cache, start_pos, indices
        )
        torch.ops.llama.update_cache_with_indices(
            k_zero_points, k_zero_points_cache, start_pos, indices
        )

        # Validate results
        self.assertTrue(torch.allclose(k_cache, self.quantized_k_cache))
        self.assertTrue(torch.allclose(k_scales_cache, self.k_scales_cache))
        self.assertTrue(torch.allclose(k_zero_points_cache, self.k_zero_points_cache))

    def test_update_kv_cache_simple(self):
        k = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        v = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        v_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)
        v_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)
        start_pos = 0
        self._update_and_validate(
            k, v, k_scales, v_scales, k_zero_points, v_zero_points, start_pos
        )

    # Tests for update_cache_with_indices functionality

    def test_basic_update_with_indices(self):
        """Test basic update with indices functionality."""
        self._reset()
        k = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 3, 8, 1), dtype=torch.int64)

        # Update positions 2, 5, 7
        indices = torch.tensor([[2, 5, 7]], dtype=torch.int64)
        start_pos = 0  # start_pos is ignored when indices are provided

        self._update_with_indices_and_validate(
            k, k_scales, k_zero_points, start_pos, indices
        )

    def test_single_index_update(self):
        """Test updating a single position with indices."""
        self._reset()
        k = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)

        # Update only position 4
        indices = torch.tensor([[4]], dtype=torch.int64)
        start_pos = 0

        self._update_with_indices_and_validate(
            k, k_scales, k_zero_points, start_pos, indices
        )

    def test_sparse_indices(self):
        """Test updating non-contiguous positions."""
        self._reset()
        k = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 3, 8, 1), dtype=torch.int64)

        # Update positions 1, 4, 8 (sparse, non-contiguous)
        indices = torch.tensor([[1, 4, 8]], dtype=torch.int64)
        start_pos = 0

        self._update_with_indices_and_validate(
            k, k_scales, k_zero_points, start_pos, indices
        )

    def test_out_of_order_indices(self):
        """Test updating positions in a non-sequential order."""
        self._reset()
        k = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 3, 8, 1), dtype=torch.int64)

        # Update positions in reverse order: 8, 5, 2
        indices = torch.tensor([[8, 5, 2]], dtype=torch.int64)
        start_pos = 0

        self._update_with_indices_and_validate(
            k, k_scales, k_zero_points, start_pos, indices
        )

    def test_indices_exceeding_cache_size(self):
        """Test behavior when indices exceed the cache size."""
        self._reset()
        k = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)

        # Try to update positions 5, 9, 15 (where 15 is out of bounds)
        indices = torch.tensor([[5, 9, 15]], dtype=torch.int64)
        start_pos = 0

        @run_in_subprocess
        def run_and_catch(k, k_cache, start_pos, indices):
            torch.ops.llama.update_cache(k, k_cache, start_pos, indices)

        exception_raised = False
        try:
            run_and_catch(k, self.quantized_k_cache, start_pos, indices)
        except Exception:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_negative_indices(self):
        """Test behavior with negative indices."""
        self._reset()
        k = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)

        # Try to update with negative indices
        indices = torch.tensor([[5, -1, 8]], dtype=torch.int64)
        start_pos = 0

        @run_in_subprocess
        def run_and_catch(k, k_cache, start_pos, indices):
            torch.ops.llama.update_cache(k, k_cache, start_pos, indices)

        exception_raised = False
        try:
            run_and_catch(k, self.quantized_k_cache, start_pos, indices)
        except Exception:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_duplicate_indices(self):
        """Test behavior when the same position is updated multiple times."""
        self._reset()
        k = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)
        v = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        v_scales = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 3, 8, 1), dtype=torch.int64)
        v_zero_points = torch.randint(0, 20, (1, 3, 8, 1), dtype=torch.int64)

        # Update with duplicate indices - the last value should be used
        indices = torch.tensor([[3, 5, 3]], dtype=torch.int64)
        start_pos = 0

        # For our reference implementation, we need to handle this case specially
        k_cache = self.quantized_k_cache.clone()
        v_cache = self.quantized_v_cache.clone()
        k_scales_cache = self.k_scales_cache.clone()
        v_scales_cache = self.v_scales_cache.clone()
        k_zero_points_cache = self.k_zero_points_cache.clone()
        v_zero_points_cache = self.v_zero_points_cache.clone()

        # Update using custom op
        torch.ops.llama.update_cache_with_indices(k, k_cache, start_pos, indices)
        torch.ops.llama.update_cache_with_indices(
            k_scales, k_scales_cache, start_pos, indices
        )
        torch.ops.llama.update_cache_with_indices(
            k_zero_points, k_zero_points_cache, start_pos, indices
        )
        torch.ops.llama.update_cache_with_indices(v, v_cache, start_pos, indices)
        torch.ops.llama.update_cache_with_indices(
            v_scales, v_scales_cache, start_pos, indices
        )
        torch.ops.llama.update_cache_with_indices(
            v_zero_points, v_zero_points_cache, start_pos, indices
        )

        # Position 3 should have the value from the last update (index 2 in the sequence)
        self.assertTrue(torch.allclose(k_cache[0, 3], k[0, 2]))
        self.assertTrue(torch.allclose(v_cache[0, 3], v[0, 2]))
        self.assertTrue(torch.allclose(k_scales_cache[0, 3], k_scales[0, 2]))
        self.assertTrue(torch.allclose(v_scales_cache[0, 3], v_scales[0, 2]))
        self.assertTrue(torch.allclose(k_zero_points_cache[0, 3], k_zero_points[0, 2]))
        self.assertTrue(torch.allclose(v_zero_points_cache[0, 3], v_zero_points[0, 2]))

        # Position 5 should have the value from index 1
        self.assertTrue(torch.allclose(k_cache[0, 5], k[0, 1]))
        self.assertTrue(torch.allclose(v_cache[0, 5], v[0, 1]))

    def test_batched_update_with_indices(self):
        """Test updating with indices in a batched setting."""
        self.batch_size = 2
        self._reset()
        k = torch.randint(0, 50, (self.batch_size, 3, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((self.batch_size, 3, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(
            0, 20, (self.batch_size, 3, 8, 1), dtype=torch.int64
        )

        # Different indices for each batch
        indices = torch.tensor(
            [[1, 4, 7], [2, 5, 8]],  # indices for batch 0  # indices for batch 1
            dtype=torch.int64,
        )
        start_pos = 0

        self._update_with_indices_and_validate(
            k, k_scales, k_zero_points, start_pos, indices
        )

    def test_different_seq_lengths_per_batch(self):
        """Test updating with different sequence lengths per batch using padding."""
        self.batch_size = 2
        self._reset()

        # Create inputs with 3 tokens
        k = torch.randint(0, 50, (self.batch_size, 3, 8, 4), dtype=torch.int8)

        # Batch 0: update 3 positions, Batch 1: update only 2 positions (use -1 as padding)
        indices = torch.tensor(
            [
                [1, 3, 5],  # 3 valid indices for batch 0
                [2, 4, -1],  # 2 valid indices for batch 1, with -1 as padding
            ],
            dtype=torch.int64,
        )
        start_pos = 0

        @run_in_subprocess
        def run_and_catch(k, k_cache, start_pos, indices):
            torch.ops.llama.update_cache(k, k_cache, start_pos, indices)

        exception_raised = False
        try:
            run_and_catch(k, self.quantized_k_cache, start_pos, indices)
        except Exception:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_update_kv_cache_large_update(self):
        self._reset()
        k = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)
        v = torch.randint(0, 50, (1, 3, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        v_scales = torch.rand((1, 3, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 3, 8, 1), dtype=torch.int64)
        v_zero_points = torch.randint(0, 20, (1, 3, 8, 1), dtype=torch.int64)
        start_pos = 0
        self._update_and_validate(
            k, v, k_scales, v_scales, k_zero_points, v_zero_points, start_pos
        )

    def test_update_kv_cache_update_nonzero_offset(self):
        self._reset()
        k = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        v = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        v_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)
        v_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)
        start_pos = 2
        self._update_and_validate(
            k, v, k_scales, v_scales, k_zero_points, v_zero_points, start_pos
        )

    def test_update_kv_cache_more_updates(self):
        self._reset()
        k = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        v = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        v_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)
        v_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)
        start_pos = 2
        self._update_and_validate(
            k, v, k_scales, v_scales, k_zero_points, v_zero_points, start_pos
        )

        k = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        v = torch.randint(0, 50, (1, 1, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        v_scales = torch.rand((1, 1, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)
        v_zero_points = torch.randint(0, 20, (1, 1, 8, 1), dtype=torch.int64)
        start_pos = 4

        self._update_and_validate(
            k, v, k_scales, v_scales, k_zero_points, v_zero_points, start_pos
        )

    def test_batched_update_kv_cache_more_updates(self):
        self.batch_size = 7
        self._reset()
        k = torch.randint(0, 50, (self.batch_size, 1, 8, 4), dtype=torch.int8)
        v = torch.randint(0, 50, (self.batch_size, 1, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((self.batch_size, 1, 8, 1), dtype=torch.float64)
        v_scales = torch.rand((self.batch_size, 1, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(
            0, 20, (self.batch_size, 1, 8, 1), dtype=torch.int64
        )
        v_zero_points = torch.randint(
            0, 20, (self.batch_size, 1, 8, 1), dtype=torch.int64
        )
        start_pos = 2
        self._update_and_validate(
            k, v, k_scales, v_scales, k_zero_points, v_zero_points, start_pos
        )

        k = torch.randint(0, 50, (self.batch_size, 1, 8, 4), dtype=torch.int8)
        v = torch.randint(0, 50, (self.batch_size, 1, 8, 4), dtype=torch.int8)
        k_scales = torch.rand((self.batch_size, 1, 8, 1), dtype=torch.float64)
        v_scales = torch.rand((self.batch_size, 1, 8, 1), dtype=torch.float64)
        k_zero_points = torch.randint(
            0, 20, (self.batch_size, 1, 8, 1), dtype=torch.int64
        )
        v_zero_points = torch.randint(
            0, 20, (self.batch_size, 1, 8, 1), dtype=torch.int64
        )
        start_pos = 4

        self._update_and_validate(
            k, v, k_scales, v_scales, k_zero_points, v_zero_points, start_pos
        )
