# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.examples.models.llama.source_transformation.attention_sink import (
    KVCacheWithAttentionSink,
)


class KVCacheWithAttentionSinkTest(unittest.TestCase):

    def _init_cache(self):
        self.kv_cache = KVCacheWithAttentionSink(
            max_batch_size=self.max_batch_size,
            window_size=self.window_size,
            sink_size=self.sink_size,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            transpose_cache=self.transpose_cache,
            dtype=self.dtype,
        )

    def setUp(self):
        torch.manual_seed(42)
        self.max_batch_size = 1
        self.window_size = 28
        self.sink_size = 4
        self.n_heads = 8
        self.head_dim = 16
        self.transpose_cache = False
        self.dtype = torch.float32
        self._init_cache()

    def test_update_empty_cache(self):
        # KV cache is empty, update will fill sink tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k = torch.ones((1, 1, 8, 16), dtype=self.dtype)
        v = torch.ones((1, 1, 8, 16), dtype=self.dtype)

        k_out, v_out = self.kv_cache.update(input_pos, k, v)

        expected_k_out = torch.cat(
            [
                torch.ones((1, 1, 8, 16), dtype=self.dtype),
                torch.zeros((1, 31, 8, 16), dtype=self.dtype),
            ],
            dim=1,
        )
        expected_v_out = torch.cat(
            [
                torch.ones((1, 1, 8, 16), dtype=self.dtype),
                torch.zeros((1, 31, 8, 16), dtype=self.dtype),
            ],
            dim=1,
        )

        torch.testing.assert_close(k_out, expected_k_out)
        torch.testing.assert_close(v_out, expected_v_out)

    def test_update_without_shift(self):
        # KV cache has enough spaces for new tokens, no shift
        input_pos = torch.tensor([0], dtype=torch.int32)
        k = torch.ones((1, 5, 8, 16), dtype=self.dtype)
        v = torch.ones((1, 5, 8, 16), dtype=self.dtype)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k = torch.full((1, 5, 8, 16), 2, dtype=self.dtype)
        v = torch.full((1, 5, 8, 16), 2, dtype=self.dtype)

        k_out, v_out = self.kv_cache.update(input_pos, k, v)

        expected_k_out = torch.cat(
            [
                torch.ones((1, 5, 8, 16), dtype=self.dtype),
                torch.full((1, 5, 8, 16), 2, dtype=self.dtype),
                torch.zeros((1, 22, 8, 16), dtype=self.dtype),
            ],
            dim=1,
        )
        expected_v_out = torch.cat(
            [
                torch.ones((1, 5, 8, 16), dtype=self.dtype),
                torch.full((1, 5, 8, 16), 2, dtype=self.dtype),
                torch.zeros((1, 22, 8, 16), dtype=self.dtype),
            ],
            dim=1,
        )

        torch.testing.assert_close(k_out, expected_k_out)
        torch.testing.assert_close(v_out, expected_v_out)

    def test_update_with_some_shift(self):
        # KV cache has some spaces for new tokens but not all, shift some tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k = torch.ones((1, 5, 8, 16), dtype=self.dtype)
        v = torch.ones((1, 5, 8, 16), dtype=self.dtype)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k = torch.full((1, 5, 8, 16), 2, dtype=self.dtype)
        v = torch.full((1, 5, 8, 16), 2, dtype=self.dtype)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([10], dtype=torch.int32)
        k = torch.full((1, 24, 8, 16), 3, dtype=self.dtype)
        v = torch.full((1, 24, 8, 16), 3, dtype=self.dtype)

        k_out, v_out = self.kv_cache.update(input_pos, k, v)

        expected_k_out = torch.cat(
            [
                torch.ones((1, 4, 8, 16), dtype=self.dtype),
                torch.full((1, 4, 8, 16), 2, dtype=self.dtype),
                torch.full((1, 24, 8, 16), 3, dtype=self.dtype),
            ],
            dim=1,
        )
        expected_v_out = torch.cat(
            [
                torch.ones((1, 4, 8, 16), dtype=self.dtype),
                torch.full((1, 4, 8, 16), 2, dtype=self.dtype),
                torch.full((1, 24, 8, 16), 3, dtype=self.dtype),
            ],
            dim=1,
        )

        torch.testing.assert_close(k_out, expected_k_out)
        torch.testing.assert_close(v_out, expected_v_out)

    def test_update_with_all_shift(self):
        # KV cache has no spaces for new tokens, shift all tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k = torch.ones((1, 5, 8, 16), dtype=self.dtype)
        v = torch.ones((1, 5, 8, 16), dtype=self.dtype)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k = torch.full((1, 28, 8, 16), 2, dtype=self.dtype)
        v = torch.full((1, 28, 8, 16), 2, dtype=self.dtype)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([33], dtype=torch.int32)
        k = torch.full((1, 6, 8, 16), 3, dtype=self.dtype)
        v = torch.full((1, 6, 8, 16), 3, dtype=self.dtype)

        k_out, v_out = self.kv_cache.update(input_pos, k, v)

        expected_k_out = torch.cat(
            [
                torch.ones((1, 4, 8, 16), dtype=self.dtype),
                torch.full((1, 22, 8, 16), 2, dtype=self.dtype),
                torch.full((1, 6, 8, 16), 3, dtype=self.dtype),
            ],
            dim=1,
        )
        expected_v_out = torch.cat(
            [
                torch.ones((1, 4, 8, 16), dtype=self.dtype),
                torch.full((1, 22, 8, 16), 2, dtype=self.dtype),
                torch.full((1, 6, 8, 16), 3, dtype=self.dtype),
            ],
            dim=1,
        )

        torch.testing.assert_close(k_out, expected_k_out)
        torch.testing.assert_close(v_out, expected_v_out)
