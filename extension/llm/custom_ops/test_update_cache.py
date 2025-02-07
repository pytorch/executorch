# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch


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
