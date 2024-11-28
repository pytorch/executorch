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
from parameterized import parameterized


class RopeWithAttentionSinkTest(unittest.TestCase):

    def _init_rope(self, params: ModelArgs, eviction_batch_size: int):
        return RopeWithAttentionSink(
            params=params,
            window_size=252,
            sink_size=4,
            eviction_batch_size=eviction_batch_size,
        )

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(
            use_kv_cache=True, enable_dynamic_shape=True, max_seq_len=256
        )
        self.rope_with_attention_sink = self._init_rope(
            params=self.params, eviction_batch_size=1
        )

    @parameterized.expand(
        [
            [0, 10, 1, 0],  # No shift
            [250, 10, 1, 246],  # Some shift
            [256, 10, 1, 246],  # All shift
            [0, 10, 30, 0],  # No shift with batch eviction
            [250, 10, 30, 220],  # Some shift with batch eviction
            [256, 10, 30, 226],  # All shift with batch eviction
        ]
    )
    def test_get_freqs(
        self, input_pos, seq_len, eviction_batch_size, expected_result_pos
    ):
        self.rope_with_attention_sink = self._init_rope(
            params=self.params, eviction_batch_size=eviction_batch_size
        )

        freqs_cos, freqs_sin = self.rope_with_attention_sink.get_freqs(
            input_pos=torch.tensor([input_pos], dtype=torch.int32),
            seq_len=seq_len,
        )

        torch.testing.assert_close(
            freqs_cos,
            self.rope_with_attention_sink.freqs_cos.narrow(
                0, expected_result_pos, seq_len
            ),
        )
        torch.testing.assert_close(
            freqs_sin,
            self.rope_with_attention_sink.freqs_sin.narrow(
                0, expected_result_pos, seq_len
            ),
        )

    @parameterized.expand(
        [
            [128, 127],  # Rotate left
            [128, 128],  # No rotation
            [128, 129],  # Rotate right
        ]
    )
    def test_rotate(self, original_position, new_position):
        seq_len = 32

        size = (1, seq_len, self.params.n_heads, self.params.head_dim)
        q = torch.rand(*size, dtype=torch.float32)
        k = torch.rand(
            *size,
            dtype=torch.float32,
        )
        freqs_cos, freqs_sin = self.rope_with_attention_sink.get_freqs(
            input_pos=torch.tensor([original_position], dtype=torch.int32),
            seq_len=seq_len,
        )
        _, pre_rotated_k = self.rope_with_attention_sink.forward(
            q=q,
            k=k,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )

        rerotated_k = self.rope_with_attention_sink.rerotate_k(
            k=pre_rotated_k,
            original_position=original_position,
            new_position=new_position,
        )

        freqs_cos, freqs_sin = self.rope_with_attention_sink.get_freqs(
            input_pos=torch.tensor([new_position], dtype=torch.int32),
            seq_len=seq_len,
        )
        _, expected_k = self.rope_with_attention_sink.forward(
            q=q,
            k=k,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )

        torch.testing.assert_close(rerotated_k, expected_k)


class KVCacheWithAttentionSinkTest(unittest.TestCase):

    _single_evict_test_cases = [
        [False, 4, 1],
        [True, 4, 1],
    ]

    _batch_evict_test_cases = [
        [False, 4, 8],
        [True, 4, 8],
    ]

    _sliding_window_test_cases = [
        [False, 0, 1],
        [True, 0, 1],
    ]

    def _init_cache(self, transpose_cache, sink_size, eviction_batch_size):
        self.params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_seq_len=self.window_size + sink_size,
        )
        self.rope_with_attention_sink = RopeWithAttentionSink(
            params=self.params,
            window_size=self.window_size,
            sink_size=sink_size,
            eviction_batch_size=eviction_batch_size,
        )
        self.kv_cache = KVCacheWithAttentionSink(
            n_heads=self.params.n_heads,
            head_dim=self.params.head_dim,
            transpose_cache=transpose_cache,
            enable_dynamic_shape=self.params.enable_dynamic_shape,
            rope=self.rope_with_attention_sink,
            max_batch_size=self.max_batch_size,
            window_size=self.window_size,
            sink_size=sink_size,
            eviction_batch_size=eviction_batch_size,
            dtype=self.dtype,
        )

    def _rand_kv_with_length(self, transpose_cache, seq_len):
        size = (
            (
                self.max_batch_size,
                seq_len,
                self.params.n_heads,
                self.params.head_dim,
            )
            if not transpose_cache
            else (
                self.max_batch_size,
                self.params.n_heads,
                seq_len,
                self.params.head_dim,
            )
        )
        if not transpose_cache:
            k = torch.rand(
                *size,
                dtype=self.dtype,
            )
            v = torch.rand(
                *size,
                dtype=self.dtype,
            )
        else:
            k = torch.rand(
                *size,
                dtype=self.dtype,
            )
            v = torch.rand(
                *size,
                dtype=self.dtype,
            )
        return k, v

    def _zero_kv_with_length(self, transpose_cache, seq_len):
        size = (
            (
                self.max_batch_size,
                seq_len,
                self.params.n_heads,
                self.params.head_dim,
            )
            if not transpose_cache
            else (
                self.max_batch_size,
                self.params.n_heads,
                seq_len,
                self.params.head_dim,
            )
        )
        if not transpose_cache:
            k = torch.zeros(
                *size,
                dtype=self.dtype,
            )
            v = torch.zeros(
                *size,
                dtype=self.dtype,
            )
        else:
            k = torch.zeros(
                *size,
                dtype=self.dtype,
            )
            v = torch.zeros(
                *size,
                dtype=self.dtype,
            )
        return k, v

    def _get_dim_to_slice(self, transpose_cache):
        return 2 if transpose_cache else 1

    def _get_expected_rotated_k(
        self, transpose_cache, k, original_position, new_position
    ):
        if transpose_cache:
            return self.rope_with_attention_sink.rerotate_k(
                k=k.transpose(1, 2),
                original_position=original_position,
                new_position=new_position,
            ).transpose(1, 2)
        else:
            return self.rope_with_attention_sink.rerotate_k(
                k=k, original_position=original_position, new_position=new_position
            )

    def setUp(self):
        torch.manual_seed(42)
        self.max_batch_size = 1
        self.window_size = 28
        self.dtype = torch.float32

    @parameterized.expand(
        _single_evict_test_cases + _batch_evict_test_cases + _sliding_window_test_cases
    )
    def test_evict_empty_cache(self, transpose_cache, sink_size, eviction_batch_size):
        self._init_cache(transpose_cache, sink_size, eviction_batch_size)

        # KV cache is empty, evict does nothing
        input_pos = torch.tensor([0], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 1) == 0

        expected_k, expected_v = self._zero_kv_with_length(
            transpose_cache, self.window_size + sink_size
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    @parameterized.expand(
        _single_evict_test_cases + _batch_evict_test_cases + _sliding_window_test_cases
    )
    def test_evict_without_shift(self, transpose_cache, sink_size, eviction_batch_size):
        dimension_to_slice = self._get_dim_to_slice(transpose_cache)

        self._init_cache(transpose_cache, sink_size, eviction_batch_size)

        # KV cache has enough spaces for new tokens, no shift
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(transpose_cache, 10)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([10], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 1) == 0

        zero_k, zero_v = self._zero_kv_with_length(
            transpose_cache, self.window_size + sink_size - 10
        )

        expected_k = torch.cat(
            [
                k,
                zero_k,
            ],
            dim=dimension_to_slice,
        )
        expected_v = torch.cat(
            [
                v,
                zero_v,
            ],
            dim=dimension_to_slice,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    @parameterized.expand(_single_evict_test_cases)
    def test_evict_with_some_shift(
        self, transpose_cache, sink_size, eviction_batch_size
    ):
        dimension_to_slice = self._get_dim_to_slice(transpose_cache)

        self._init_cache(transpose_cache, sink_size, eviction_batch_size)

        # KV cache has some spaces for new tokens but not all, shift some tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(transpose_cache, 5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(transpose_cache, 5)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([10], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 24) == -2

        zero_k, zero_v = self._zero_kv_with_length(transpose_cache, 24)
        expected_k = torch.cat(
            [
                k.narrow(dimension_to_slice, 0, sink_size),
                self._get_expected_rotated_k(
                    transpose_cache, k1.narrow(dimension_to_slice, 1, 4), 6, 4
                ),
                zero_k,
            ],
            dim=dimension_to_slice,
        )
        expected_v = torch.cat(
            [
                v.narrow(dimension_to_slice, 0, sink_size),
                v1.narrow(dimension_to_slice, 1, 4),
                zero_v,
            ],
            dim=dimension_to_slice,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    @parameterized.expand(_single_evict_test_cases)
    def test_evict_with_all_shift(
        self, transpose_cache, sink_size, eviction_batch_size
    ):
        dimension_to_slice = self._get_dim_to_slice(transpose_cache)

        self._init_cache(transpose_cache, sink_size, eviction_batch_size)

        # KV cache has no spaces for new tokens, shift all tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(transpose_cache, 5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(transpose_cache, 27)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([32], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 6) == -6

        zero_k, zero_v = self._zero_kv_with_length(transpose_cache, 6)
        expected_k = torch.cat(
            [
                k.narrow(dimension_to_slice, 0, sink_size),
                self._get_expected_rotated_k(
                    transpose_cache, k1.narrow(dimension_to_slice, 5, 22), 10, 4
                ),
                zero_k,
            ],
            dim=dimension_to_slice,
        )
        expected_v = torch.cat(
            [
                v.narrow(dimension_to_slice, 0, sink_size),
                v1.narrow(dimension_to_slice, 5, 22),
                zero_v,
            ],
            dim=dimension_to_slice,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    @parameterized.expand(_sliding_window_test_cases)
    def test_evict_with_some_shift_for_sliding_window(
        self, transpose_cache, sink_size, eviction_batch_size
    ):
        dimension_to_slice = self._get_dim_to_slice(transpose_cache)

        self._init_cache(transpose_cache, sink_size, eviction_batch_size)

        # KV cache has some spaces for new tokens but not all, shift some tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(transpose_cache, 5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(transpose_cache, 5)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([10], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 20) == -2

        zero_k, zero_v = self._zero_kv_with_length(transpose_cache, 20)
        expected_k = torch.cat(
            [
                self._get_expected_rotated_k(
                    transpose_cache, k.narrow(dimension_to_slice, 2, 3), 2, 0
                ),
                self._get_expected_rotated_k(transpose_cache, k1, 5, 3),
                zero_k,
            ],
            dim=dimension_to_slice,
        )
        expected_v = torch.cat(
            [
                v.narrow(dimension_to_slice, 2, 3),
                v1,
                zero_v,
            ],
            dim=dimension_to_slice,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    @parameterized.expand(_sliding_window_test_cases)
    def test_evict_with_all_shift_for_sliding_window(
        self, transpose_cache, sink_size, eviction_batch_size
    ):
        dimension_to_slice = self._get_dim_to_slice(transpose_cache)

        self._init_cache(transpose_cache, sink_size, eviction_batch_size)

        # KV cache has no spaces for new tokens, shift all tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(transpose_cache, 5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(transpose_cache, 23)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([28], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 6) == -6

        zero_k, zero_v = self._zero_kv_with_length(transpose_cache, 6)
        expected_k = torch.cat(
            [
                self._get_expected_rotated_k(
                    transpose_cache, k1.narrow(dimension_to_slice, 1, 22), 6, 0
                ),
                zero_k,
            ],
            dim=dimension_to_slice,
        )
        expected_v = torch.cat(
            [
                v1.narrow(dimension_to_slice, 1, 22),
                zero_v,
            ],
            dim=dimension_to_slice,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    @parameterized.expand(_batch_evict_test_cases)
    def test_batch_evict_with_seq_len(
        self, transpose_cache, sink_size, eviction_batch_size
    ):
        dimension_to_slice = self._get_dim_to_slice(transpose_cache)

        self._init_cache(transpose_cache, sink_size, eviction_batch_size)

        # KV cache has some spaces for new tokens but not all, shift some tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(transpose_cache, 5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(transpose_cache, 25)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([30], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 12) == -10

        zero_k, zero_v = self._zero_kv_with_length(transpose_cache, 12)
        expected_k = torch.cat(
            [
                k.narrow(dimension_to_slice, 0, sink_size),
                self._get_expected_rotated_k(
                    transpose_cache, k1.narrow(dimension_to_slice, 9, 16), 14, 4
                ),
                zero_k,
            ],
            dim=dimension_to_slice,
        )
        expected_v = torch.cat(
            [
                v.narrow(dimension_to_slice, 0, sink_size),
                v1.narrow(dimension_to_slice, 9, 16),
                zero_v,
            ],
            dim=dimension_to_slice,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)

    @parameterized.expand(_batch_evict_test_cases)
    def test_batch_evict_with_batch_size(
        self, transpose_cache, sink_size, eviction_batch_size
    ):
        dimension_to_slice = self._get_dim_to_slice(transpose_cache)

        self._init_cache(transpose_cache, sink_size, eviction_batch_size)

        # KV cache has no spaces for new tokens, shift all tokens
        input_pos = torch.tensor([0], dtype=torch.int32)
        k, v = self._rand_kv_with_length(transpose_cache, 5)

        self.kv_cache.update(input_pos, k, v)

        input_pos = torch.tensor([5], dtype=torch.int32)
        k1, v1 = self._rand_kv_with_length(transpose_cache, 25)

        self.kv_cache.update(input_pos, k1, v1)

        input_pos = torch.tensor([30], dtype=torch.int32)
        assert self.kv_cache.evict_tokens(input_pos, 6) == -8

        zero_k, zero_v = self._zero_kv_with_length(transpose_cache, 10)
        expected_k = torch.cat(
            [
                k.narrow(dimension_to_slice, 0, sink_size),
                self._get_expected_rotated_k(
                    transpose_cache, k1.narrow(dimension_to_slice, 7, 18), 12, 4
                ),
                zero_k,
            ],
            dim=dimension_to_slice,
        )
        expected_v = torch.cat(
            [
                v.narrow(dimension_to_slice, 0, sink_size),
                v1.narrow(dimension_to_slice, 7, 18),
                zero_v,
            ],
            dim=dimension_to_slice,
        )

        torch.testing.assert_close(self.kv_cache.k_cache, expected_k)
        torch.testing.assert_close(self.kv_cache.v_cache, expected_v)
