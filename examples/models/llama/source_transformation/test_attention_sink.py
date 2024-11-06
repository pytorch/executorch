# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.llama.llama_transformer import ModelArgs

from executorch.examples.models.llama.source_transformation.attention_sink import (
    KVCacheWithAttentionSink,
    RopeWithAttentionSink,
)


class RopeWithAttentionSinkTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(use_kv_cache=True, enable_dynamic_shape=True)
        self.rope_with_attention_sink = RopeWithAttentionSink(params=self.params)
        self.seq_len = 32
        self.n_local_heads = self.params.n_heads
        self.n_local_kv_heads = self.params.n_heads
        self.head_dim = self.params.dim // self.params.n_heads
        self.q = torch.rand(
            1, self.seq_len, self.n_local_heads, self.head_dim, dtype=torch.float32
        )
        self.k = torch.rand(
            1,
            self.seq_len,
            self.n_local_kv_heads,
            self.head_dim,
            dtype=torch.float32,
        )

    def test_rotate_backward(self):
        original_position = 128
        new_position = 127

        _, pre_rotated_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([original_position], dtype=torch.int32),
        )

        k = self.rope_with_attention_sink.rerotate_k(
            k=pre_rotated_k,
            original_position=original_position,
            new_position=new_position,
        )

        _, expected_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([new_position], dtype=torch.int32),
        )

        torch.testing.assert_close(k, expected_k)

    def test_rotate_inplace(self):
        original_position = 128
        new_position = 128

        _, pre_rotated_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([original_position], dtype=torch.int32),
        )

        k = self.rope_with_attention_sink.rerotate_k(
            k=pre_rotated_k,
            original_position=original_position,
            new_position=new_position,
        )

        _, expected_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([new_position], dtype=torch.int32),
        )

        torch.testing.assert_close(k, expected_k)

    def test_rotate_forward(self):
        original_position = 128
        new_position = 129

        _, pre_rotated_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([original_position], dtype=torch.int32),
        )

        k = self.rope_with_attention_sink.rerotate_k(
            k=pre_rotated_k,
            original_position=original_position,
            new_position=new_position,
        )

        _, expected_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([new_position], dtype=torch.int32),
        )

        torch.testing.assert_close(k, expected_k)


class KVCacheWithAttentionSinkTest(unittest.TestCase):

    def _init_cache(self):
        self.kv_cache = KVCacheWithAttentionSink(
            n_heads=self.params.n_heads,
            head_dim=self.params.dim // self.params.n_heads,
            transpose_cache=False,
            enable_dynamic_shape=self.params.enable_dynamic_shape,
            rope=self.rope_with_attention_sink,
            max_batch_size=self.max_batch_size,
            window_size=self.window_size,
            sink_size=self.sink_size,
            dtype=self.dtype,
        )

    def _init_cache_with_batch_size(self):
        self.kv_cache = KVCacheWithAttentionSink(
            n_heads=self.params.n_heads,
            head_dim=self.params.dim // self.params.n_heads,
            transpose_cache=False,
            enable_dynamic_shape=self.params.enable_dynamic_shape,
            rope=self.rope_with_attention_sink,
            max_batch_size=self.max_batch_size,
            window_size=self.window_size,
            sink_size=self.sink_size,
            eviction_batch_size=self.eviction_batch_size,
            dtype=self.dtype,
        )

    def _rand_kv_with_length(self, seq_len):
        k = torch.rand(
            self.max_batch_size,
            seq_len,
            self.params.n_heads,
            self.params.dim // self.params.n_heads,
            dtype=self.dtype,
        )
        v = torch.rand(
            self.max_batch_size,
            seq_len,
            self.params.n_heads,
            self.params.dim // self.params.n_heads,
            dtype=self.dtype,
        )
        return k, v

    def _zero_kv_with_length(self, seq_len):
        k = torch.zeros(
            (
                self.max_batch_size,
                seq_len,
                self.params.n_heads,
                self.params.dim // self.params.n_heads,
            ),
            dtype=self.dtype,
        )
        v = torch.zeros(
            (
                self.max_batch_size,
                seq_len,
                self.params.n_heads,
                self.params.dim // self.params.n_heads,
            ),
            dtype=self.dtype,
        )
        return k, v

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(use_kv_cache=True, enable_dynamic_shape=True)
        self.rope_with_attention_sink = RopeWithAttentionSink(params=self.params)
        self.max_batch_size = 1
        self.window_size = 28
        self.sink_size = 4
        self.dtype = torch.float32
        self.eviction_batch_size = 8

    def test_evict_empty_cache(self):
        self._init_cache()

        # KV cache is empty, evict does nothing
        input_pos = torch.tensor([0], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 1) == 0

        expected_k, expected_v = self._zero_kv_with_length(
            self.window_size + self.sink_size
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    def test_evict_without_shift(self):
        self._init_cache()

        # KV cache has enough spaces for new tokens, no shift
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(10)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([10], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 1) == 0

        zero_k, zero_v = self._zero_kv_with_length(22)

        expected_k = torch.cat(
            [
                k,
                zero_k,
            ],
            dim=1,
        )
        expected_v = torch.cat(
            [
                v,
                zero_v,
            ],
            dim=1,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    def test_evict_with_some_shift(self):
        self._init_cache()

        # KV cache has some spaces for new tokens but not all, shift some tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(5)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([10], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 24) == -2

        zero_k, zero_v = self._zero_kv_with_length(24)
        expected_k = torch.cat(
            [
                k[:, :4, :, :],
                self.rope_with_attention_sink.rerotate_k(k1[:, 1:, :, :], 6, 4),
                zero_k,
            ],
            dim=1,
        )
        expected_v = torch.cat(
            [
                v[:, :4, :, :],
                v1[:, 1:, :, :],
                zero_v,
            ],
            dim=1,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    def test_evict_with_all_shift(self):
        self._init_cache()

        # KV cache has no spaces for new tokens, shift all tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(27)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([32], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 6) == -6

        zero_k, zero_v = self._zero_kv_with_length(6)
        expected_k = torch.cat(
            [
                k[:, :4, :, :],
                self.rope_with_attention_sink.rerotate_k(k1[:, 5:, :, :], 10, 4),
                zero_k,
            ],
            dim=1,
        )
        expected_v = torch.cat(
            [
                v[:, :4, :, :],
                v1[:, 5:, :, :],
                zero_v,
            ],
            dim=1,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    def test_batch_evict_empty_cache(self):
        self._init_cache_with_batch_size()

        # KV cache is empty, evict does nothing
        input_pos = torch.tensor([0], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 1) == 0

        expected_k, expected_v = self._zero_kv_with_length(
            self.window_size + self.sink_size
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    def test_batch_evict_without_shift(self):
        self._init_cache_with_batch_size()

        # KV cache has enough spaces for new tokens, no shift
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(10)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([10], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 1) == 0

        zero_k, zero_v = self._zero_kv_with_length(22)

        expected_k = torch.cat(
            [
                k,
                zero_k,
            ],
            dim=1,
        )
        expected_v = torch.cat(
            [
                v,
                zero_v,
            ],
            dim=1,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    def test_batch_evict_with_seq_len(self):
        self._init_cache_with_batch_size()

        # KV cache has some spaces for new tokens but not all, shift some tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(25)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([30], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 12) == -10

        zero_k, zero_v = self._zero_kv_with_length(12)
        expected_k = torch.cat(
            [
                k[:, :4, :, :],
                self.rope_with_attention_sink.rerotate_k(k1[:, 9:, :, :], 14, 4),
                zero_k,
            ],
            dim=1,
        )
        expected_v = torch.cat(
            [
                v[:, :4, :, :],
                v1[:, 9:, :, :],
                zero_v,
            ],
            dim=1,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    def test_batch_evict_with_batch_size(self):
        self._init_cache_with_batch_size()

        # KV cache has no spaces for new tokens, shift all tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(25)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([30], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 6) == -8

        zero_k, zero_v = self._zero_kv_with_length(10)
        expected_k = torch.cat(
            [
                k[:, :4, :, :],
                self.rope_with_attention_sink.rerotate_k(k1[:, 7:, :, :], 12, 4),
                zero_k,
            ],
            dim=1,
        )
        expected_v = torch.cat(
            [
                v[:, :4, :, :],
                v1[:, 7:, :, :],
                zero_v,
            ],
            dim=1,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)
