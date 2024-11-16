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

from executorch.examples.models.llama.source_transformation.sdpa import SDPACustom


class SDPAWithQuantizedKVCacheTest(unittest.TestCase):

    def _init_cache(self):
        self.kv_cache = KVCache(
            self.max_batch_size,
            self.max_seq_len,
            self.n_kv_heads,
            self.head_dim,
            False,
            self.enable_dynamic_shape,
            dtype=self.dtype,
        )
        self.quantized_kv_cache = QuantizedKVCache.from_float(
            self.kv_cache, QuantizedCacheType.AffineAsymmetric
        )

    def _init_kv(self):
        kv_shape = (1, self.seq_len, self.n_kv_heads, self.head_dim)
        q_shape = (1, self.seq_len, self.n_heads, self.head_dim)
        q = torch.rand(q_shape, dtype=self.dtype)
        k = torch.rand(kv_shape, dtype=self.dtype)
        v = torch.rand(kv_shape, dtype=self.dtype)
        return q, k, v

    def setUp(self):
        torch.manual_seed(42)
        self.max_batch_size = 1
        self.max_seq_len = 5
        self.n_kv_heads = 4
        self.n_heads = 8
        self.head_dim = 17
        self.dim = self.n_heads * self.head_dim
        self.enable_dynamic_shape = False
        self.dtype = torch.float32

    def test_simple(self, is_dynamic_shape=False):
        self.enable_dynamic_shape = is_dynamic_shape
        input_pos = torch.tensor([0], dtype=torch.int64)
        self.seq_len = 3
        self._init_cache()
        q, k, v = self._init_kv()
        self.float_sdpa = SDPACustom(self.kv_cache, self.dim)
        self.quantized_sdpa = SDPACustom(self.quantized_kv_cache, self.dim)
        float_out = self.float_sdpa(input_pos, q, k, v, 1, self.seq_len, None)
        quantized_out = self.quantized_sdpa(input_pos, q, k, v, 1, self.seq_len, None)
        torch.testing.assert_close(
            float_out,
            quantized_out,
        )

        input_pos = torch.tensor([3], dtype=torch.int64)
        self.seq_len = 1
        q, k, v = self._init_kv()
        float_out = self.float_sdpa(input_pos, q, k, v, 1, self.seq_len, None)
        quantized_out = self.quantized_sdpa(input_pos, q, k, v, 1, self.seq_len, None)
        torch.testing.assert_close(
            float_out,
            quantized_out,
            rtol=1e-03,
            atol=1e-03,
        )
