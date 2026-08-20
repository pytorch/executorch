# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.examples.models.llama.attention import KVCache

from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    CustomKVCache,
    QuantizedCacheType,
    QuantizedKVCache,
)

from executorch.examples.models.llama.source_transformation.sdpa import SDPACustom


class SDPAWithQuantizedKVCacheTest(unittest.TestCase):
    def _init_cache(self):
        self.kv_cache = KVCache(
            self.max_batch_size,
            self.max_context_len,
            self.n_kv_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            dtype=self.dtype,
        )
        self.quantized_kv_cache = QuantizedKVCache.from_float(
            self.kv_cache, QuantizedCacheType.AffineAsymmetric, True
        )
        # Need this because first test actually has seq_len > 1
        # and vanilla kvcache cannot handle seq_len > 1, due to
        # how input_pos encoding works in the current stack.
        # This needs fixing by making sure rest of the stack including
        # custom ops or other backends can work with input_pos
        # as a sequence of token positions
        self.custom_kv_cache = CustomKVCache(
            self.max_batch_size,
            self.max_context_len,
            self.n_kv_heads,
            self.head_dim,
            dtype=self.dtype,
        )

    def _init_kv(self):
        kv_shape = (1, self.n_kv_heads, self.seq_len, self.head_dim)
        q_shape = (1, self.n_heads, self.seq_len, self.head_dim)
        q = torch.rand(q_shape, dtype=self.dtype)
        k = torch.rand(kv_shape, dtype=self.dtype)
        v = torch.rand(kv_shape, dtype=self.dtype)
        return q, k, v

    def setUp(self):
        torch.manual_seed(42)
        self.max_batch_size = 1
        self.max_context_len = 5
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
        q, k_val, v_val = self._init_kv()
        self.float_sdpa = SDPACustom(self.dim)
        self.quantized_sdpa = SDPACustom(self.dim)
        k, v = self.custom_kv_cache.update(input_pos, k_val, v_val)
        float_out = self.float_sdpa(input_pos, q, k, v, 1, self.seq_len, None)
        k, v = self.quantized_kv_cache.update(input_pos, k_val, v_val)
        quantized_out = self.quantized_sdpa(input_pos, q, k, v, 1, self.seq_len, None)
        torch.testing.assert_close(
            float_out,
            quantized_out,
        )

        input_pos = torch.tensor([3], dtype=torch.int64)
        self.seq_len = 1
        q, k_val, v_val = self._init_kv()
        k, v = self.custom_kv_cache.update(input_pos, k_val, v_val)
        float_out = self.float_sdpa(input_pos, q, k, v, 1, self.seq_len, None)
        k, v = self.quantized_kv_cache.update(input_pos, k_val, v_val)
        quantized_out = self.quantized_sdpa(input_pos, q, k, v, 1, self.seq_len, None)
        torch.testing.assert_close(
            float_out,
            quantized_out,
            rtol=1e-03,
            atol=1e-03,
        )
