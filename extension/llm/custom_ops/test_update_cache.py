# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import multiprocessing
import sys
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


class ChannelwiseGatedDeltaRuleTest(unittest.TestCase):
    def _make_inputs(
        self,
        batch_size: int = 2,
        num_heads: int = 3,
        seq_len: int = 4,
        k_head_dim: int = 5,
        v_head_dim: int = 6,
    ):
        query = torch.randn(batch_size, num_heads, seq_len, k_head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, k_head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, v_head_dim)
        # Per-key-channel decay, passed already exponentiated (in (0, 1)).
        decay = torch.rand(batch_size, num_heads, seq_len, k_head_dim)
        beta = torch.sigmoid(torch.randn(batch_size, num_heads, seq_len))
        initial_state = torch.randn(batch_size, num_heads, k_head_dim, v_head_dim)
        return query, key, value, decay, beta, initial_state

    def _reference_channelwise_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ):
        state = initial_state.clone()
        output = torch.zeros_like(value)

        for token in range(query.size(2)):
            # Per-key-channel decay: [B, H, K, 1], already exponentiated.
            decay_t = decay[:, :, token].unsqueeze(-1)
            beta_t = beta[:, :, token].unsqueeze(-1)
            k_t = key[:, :, token]
            v_t = value[:, :, token]
            q_t = query[:, :, token]

            state = state * decay_t
            v_pred = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - v_pred) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            output[:, :, token] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

        return output, state

    def test_channelwise_gated_delta_rule_matches_reference(self):
        torch.manual_seed(0)

        test_cases = (
            (2, 3, 4, 5, 6),
            (1, 4, 7, 8, 3),
        )

        for case in test_cases:
            with self.subTest(case=case):
                (
                    query,
                    key,
                    value,
                    decay,
                    beta,
                    initial_state,
                ) = self._make_inputs(*case)

                expected_output, expected_state = (
                    self._reference_channelwise_gated_delta_rule(
                        query,
                        key,
                        value,
                        decay,
                        beta,
                        initial_state,
                    )
                )

                # Functional op: initial_state must not be mutated.
                initial_state_before = initial_state.clone()
                actual_output, actual_state = (
                    torch.ops.llama.channelwise_gated_delta_rule(
                        query,
                        key,
                        value,
                        decay,
                        beta,
                        initial_state,
                    )
                )

                self.assertTrue(
                    torch.allclose(actual_output, expected_output, atol=1e-5)
                )
                self.assertTrue(torch.allclose(actual_state, expected_state, atol=1e-5))
                self.assertTrue(torch.equal(initial_state, initial_state_before))

    def test_channelwise_gated_delta_rule_out_matches_reference(self):
        torch.manual_seed(0)

        query, key, value, decay, beta, initial_state = self._make_inputs()
        expected_output, expected_state = self._reference_channelwise_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            initial_state,
        )

        actual_output = torch.empty_like(value)
        actual_final_state = torch.empty_like(initial_state)
        returned_output, returned_state = (
            torch.ops.llama.channelwise_gated_delta_rule.out(
                query,
                key,
                value,
                decay,
                beta,
                initial_state,
                out=actual_output,
                final_state_out=actual_final_state,
            )
        )

        self.assertEqual(returned_output.data_ptr(), actual_output.data_ptr())
        self.assertEqual(returned_state.data_ptr(), actual_final_state.data_ptr())
        self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-5))
        self.assertTrue(torch.allclose(actual_final_state, expected_state, atol=1e-5))

    def test_channelwise_gated_delta_rule_out_invalid_args_raise(
        self,
    ):
        torch.manual_seed(0)

        query, key, value, decay, beta, initial_state = self._make_inputs()
        invalid_key = key[:, :, :, :-1].contiguous()
        actual_output = torch.empty(1)
        actual_final_state = torch.empty(1)

        with self.assertRaises(RuntimeError):
            torch.ops.llama.channelwise_gated_delta_rule.out(
                query,
                invalid_key,
                value,
                decay,
                beta,
                initial_state,
                out=actual_output,
                final_state_out=actual_final_state,
            )

        self.assertEqual(tuple(actual_output.shape), (1,))
        self.assertEqual(tuple(actual_final_state.shape), (1,))

    def test_channelwise_gated_delta_rule_out_rejects_initial_state_alias(
        self,
    ):
        torch.manual_seed(0)

        query, key, value, decay, beta, initial_state = self._make_inputs()
        actual_output = torch.empty_like(value)

        with self.assertRaisesRegex(
            RuntimeError,
            "final_state_out must not alias initial_state",
        ):
            torch.ops.llama.channelwise_gated_delta_rule.out(
                query,
                key,
                value,
                decay,
                beta,
                initial_state,
                out=actual_output,
                final_state_out=initial_state,
            )

    def test_channelwise_gated_delta_rule_out_rejects_initial_state_overlap(
        self,
    ):
        torch.manual_seed(0)

        query, key, value, decay, beta, initial_state = self._make_inputs(
            batch_size=1,
            num_heads=1,
            k_head_dim=2,
            v_head_dim=3,
        )
        actual_output = torch.empty_like(value)
        final_state_out = initial_state.as_strided(
            (1, 1, 1, 5),
            (5, 5, 5, 1),
            storage_offset=1,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "final_state_out must not alias initial_state",
        ):
            torch.ops.llama.channelwise_gated_delta_rule.out(
                query,
                key,
                value,
                decay,
                beta,
                initial_state,
                out=actual_output,
                final_state_out=final_state_out,
            )

    def test_channelwise_gated_delta_rule_chunked_matches_full_sequence(self):
        torch.manual_seed(0)

        # seq_len > CHUNK_SIZE (32 in the C++ kernel): the single full call runs
        # the internal chunk loop over multiple chunks (6 * 32 + 8), while the
        # per-segment calls are chunked externally as (0, 64), (64, 192),
        # (192, 200) — 2, 4, and 1 partial internal chunks respectively — at
        # different boundaries than the single-call internal loop. Equality
        # validates the inter-chunk state carry independent of outer
        # segmentation.
        # fp32 accumulation over a 32-token chunk needs a looser tol than the
        # single-chunk tests above.
        seq_len = 200
        query, key, value, decay, beta, initial_state = self._make_inputs(
            seq_len=seq_len
        )

        full_output, full_state = torch.ops.llama.channelwise_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            initial_state,
        )

        chunk_state = initial_state
        chunk_outputs = []
        for start, end in ((0, 64), (64, 192), (192, 200)):
            chunk_output, chunk_state = torch.ops.llama.channelwise_gated_delta_rule(
                query[:, :, start:end, :],
                key[:, :, start:end, :],
                value[:, :, start:end, :],
                decay[:, :, start:end, :],
                beta[:, :, start:end],
                chunk_state,
            )
            chunk_outputs.append(chunk_output)

        chunked_output = torch.cat(chunk_outputs, dim=2)
        self.assertTrue(
            torch.allclose(chunked_output, full_output, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(torch.allclose(chunk_state, full_state, atol=1e-3, rtol=1e-3))

    def test_channelwise_gated_delta_rule_multichunk_matches_reference(self):
        torch.manual_seed(0)

        # Drive the prefill route past a single CHUNK_SIZE (32) so the kernel's
        # internal chunk loop runs multiple times against the token-by-token
        # reference: 130 exercises a ragged final chunk (4 * 32 + 2), 256
        # exercises an exact multiple (8 * 32). fp32 accumulation over a full
        # chunk needs a looser tol than the single-chunk tests.
        for seq_len in (130, 256):
            with self.subTest(seq_len=seq_len):
                inputs = self._make_inputs(seq_len=seq_len)
                expected_output, expected_state = (
                    self._reference_channelwise_gated_delta_rule(*inputs)
                )

                actual_output, actual_state = (
                    torch.ops.llama.channelwise_gated_delta_rule(*inputs)
                )

                self.assertTrue(
                    torch.allclose(actual_output, expected_output, atol=1e-3, rtol=1e-3)
                )
                self.assertTrue(
                    torch.allclose(actual_state, expected_state, atol=1e-3, rtol=1e-3)
                )

    def test_channelwise_gated_delta_rule_exports(self):
        class Module(torch.nn.Module):
            def forward(self, query, key, value, decay, beta, initial_state):
                return torch.ops.llama.channelwise_gated_delta_rule(
                    query, key, value, decay, beta, initial_state
                )

        inputs = self._make_inputs()

        # Static export: the op must survive as a single graph node (the Meta
        # impl lets it trace without running the real kernel).
        ep = torch.export.export(Module(), inputs)
        targets = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
        self.assertIn("llama.channelwise_gated_delta_rule.default", targets)

        # Dynamic sequence length: one graph shared across prefill/decode.
        seq = torch.export.Dim("seq", min=1, max=128)
        dynamic_shapes = (
            {2: seq},  # query          [B, H, T, K]
            {2: seq},  # key            [B, H, T, K]
            {2: seq},  # value          [B, H, T, V]
            {2: seq},  # decay          [B, H, T, K]
            {2: seq},  # beta           [B, H, T]
            {},  # initial_state  [B, H, K, V] (no T dim)
        )
        ep_dyn = torch.export.export(Module(), inputs, dynamic_shapes=dynamic_shapes)
        self.assertTrue(
            any(
                "channelwise_gated_delta_rule" in str(n.target)
                for n in ep_dyn.graph.nodes
                if n.op == "call_function"
            )
        )

    @unittest.skipUnless(
        sys.platform == "linux",
        "Custom-kernel .pte execution via the Python Runtime is Linux-only: the "
        "channelwise_gated_delta_rule kernel is not registered in the ExecuTorch "
        "pybindings runtime on Windows (custom-op static registration does not "
        "cross the DLL boundary). Eager + export paths are covered on all "
        "platforms by the other tests.",
    )
    def test_channelwise_gated_delta_rule_pte_execution(self):
        # Exports, lowers, and *executes* the op through the ExecuTorch runtime.
        # This is the only test that hits the boxed kernel and its stack
        # arg-count contract: the emitter appends a trailing TensorList to a
        # multi-output out variant, so the runtime passes 9 args, not 8. Eager
        # and export-trace tests never exercise that path.
        from executorch.exir import EdgeCompileConfig, to_edge
        from executorch.runtime import Runtime

        class Module(torch.nn.Module):
            def forward(self, query, key, value, decay, beta, initial_state):
                return torch.ops.llama.channelwise_gated_delta_rule(
                    query, key, value, decay, beta, initial_state
                )

        runtime = Runtime.get()
        for seq_len in (4, 1):  # T != 1 (chunked route) and T == 1 (decode route)
            with self.subTest(seq_len=seq_len):
                torch.manual_seed(seq_len)
                inputs = self._make_inputs(seq_len=seq_len)
                expected_output, expected_state = (
                    self._reference_channelwise_gated_delta_rule(*inputs)
                )

                ep = torch.export.export(Module(), inputs)
                edge = to_edge(
                    ep, compile_config=EdgeCompileConfig(_check_ir_validity=False)
                )
                program_buffer = edge.to_executorch().buffer

                program = runtime.load_program(program_buffer)
                method = program.load_method("forward")
                output, final_state = method.execute(list(inputs))

                self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))
                self.assertTrue(torch.allclose(final_state, expected_state, atol=1e-5))
