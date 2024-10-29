# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.examples.models.llama.llama_transformer import KVCache

from executorch.examples.models.llama.source_transformation.quantized_kv_cache import (
    QuantizedCacheType,
    QuantizedKVCache,
)


class QuantizedKVCacheTest(unittest.TestCase):

    def _init_cache(self):
        self.kv_cache = KVCache(
            self.max_batch_size,
            self.max_seq_len,
            self.n_kv_heads,
            self.head_dim,
            self.transpose_kv_cache,
            self.enable_dynamic_shape,
            dtype=self.dtype,
        )

    def _init_kv(self):
        if self.transpose_kv_cache:
            shape = (1, self.n_kv_heads, self.seq_len, self.head_dim)
        else:
            shape = (1, self.seq_len, self.n_kv_heads, self.head_dim)
        k = torch.rand(shape, dtype=self.dtype)
        v = torch.rand(shape, dtype=self.dtype)
        return k, v

    def setUp(self):
        torch.manual_seed(42)
        self.max_batch_size = 1
        self.max_seq_len = 5
        self.n_kv_heads = 8
        self.head_dim = 17
        self.enable_dynamic_shape = False
        self.transpose_kv_cache = False
        self.dtype = torch.float32

    def _test_simple_update_fetch(self, is_transposed=False, is_dynamic_shape=False):
        self.transpose_kv_cache = is_transposed
        self.enable_dynamic_shape = is_dynamic_shape
        input_pos = torch.tensor([0, 1, 2])
        self.seq_len = input_pos.size(0)
        self._init_cache()
        k, v = self._init_kv()
        quantized_kv_cache = QuantizedKVCache.from_float(
            self.kv_cache, QuantizedCacheType.AffineAsymmetric
        )
        updated_k_cache, updated_v_cache = self.kv_cache.update(input_pos, k, v)
        updated_dequantized_k_cache, updated_dequantized_v_cache = (
            quantized_kv_cache.update(input_pos, k, v)
        )

        def index(t, input_pos):
            if self.transpose_kv_cache:
                return t[:, :, input_pos, :]
            else:
                return t[:, input_pos, :, :]

        sliced_k_cache = index(updated_k_cache, input_pos)
        sliced_v_cache = index(updated_v_cache, input_pos)

        sliced_dequantized_k_cache = index(updated_dequantized_k_cache, input_pos)
        sliced_dequantized_v_cache = index(updated_dequantized_v_cache, input_pos)

        torch.testing.assert_close(
            sliced_k_cache,
            sliced_dequantized_k_cache,
            rtol=1e-02,
            atol=1e-02,
        )
        torch.testing.assert_close(
            sliced_v_cache,
            sliced_dequantized_v_cache,
            rtol=1e-02,
            atol=1e-02,
        )

        input_pos = torch.tensor([3])
        self.seq_len = input_pos.size(0)
        k, v = self._init_kv()
        pos_to_check = torch.tensor([0, 1, 2, 3])
        updated_k_cache, updated_v_cache = self.kv_cache.update(input_pos, k, v)
        updated_dequantized_k_cache, updated_dequantized_v_cache = (
            quantized_kv_cache.update(input_pos, k, v)
        )
        sliced_k_cache = index(updated_k_cache, pos_to_check)
        sliced_v_cache = index(updated_v_cache, pos_to_check)

        sliced_dequantized_k_cache = index(updated_dequantized_k_cache, pos_to_check)
        sliced_dequantized_v_cache = index(updated_dequantized_v_cache, pos_to_check)

        torch.testing.assert_close(
            sliced_k_cache,
            sliced_dequantized_k_cache,
            rtol=1e-02,
            atol=1e-02,
        )
        torch.testing.assert_close(
            sliced_v_cache,
            sliced_dequantized_v_cache,
            rtol=1e-02,
            atol=1e-02,
        )

    def test_simple_update_fetch_not_transposed(self):
        self._test_simple_update_fetch()

    def test_simple_update_fetch_not_transposed_dynamic_shape(self):
        self._test_simple_update_fetch(is_dynamic_shape=True)

    def test_simple_update_fetch_transposed(self):
        self._test_simple_update_fetch(is_transposed=True)

    def test_simple_update_fetch_transposed_dynamic_shape(self):
        self._test_simple_update_fetch(is_transposed=True, is_dynamic_shape=True)
