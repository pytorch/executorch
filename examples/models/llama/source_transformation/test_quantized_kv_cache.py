# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.examples.models.llama.attention import KVCache

from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    QuantizedCacheType,
    QuantizedKVCache,
)


class QuantizedKVCacheTest(unittest.TestCase):
    def _init_cache(self):
        self.kv_cache = KVCache(
            self.max_batch_size,
            self.max_context_len,
            self.n_kv_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            dtype=self.dtype,
        )

    def _init_kv(self):
        shape = (1, self.n_kv_heads, self.seq_len, self.head_dim)
        k = torch.rand(shape, dtype=self.dtype)
        v = torch.rand(shape, dtype=self.dtype)
        return k, v

    def setUp(self):
        torch.manual_seed(42)
        self.max_batch_size = 1
        self.max_context_len = 5
        self.n_kv_heads = 8
        self.head_dim = 17
        self.enable_dynamic_shape = False
        self.dtype = torch.float32

    def _test_simple_update_fetch(
        self, is_dynamic_shape=False, use_custom_update_cache_op=False
    ):
        self.enable_dynamic_shape = is_dynamic_shape
        input_pos = torch.tensor([0, 1, 2])
        self.seq_len = input_pos.size(0)
        self._init_cache()
        k, v = self._init_kv()
        quantized_kv_cache = QuantizedKVCache.from_float(
            self.kv_cache,
            QuantizedCacheType.AffineAsymmetric,
            use_custom_update_cache_op,
        )
        updated_k_cache, updated_v_cache = self.kv_cache.update(input_pos, k, v)
        (
            updated_dequantized_k_cache,
            updated_dequantized_v_cache,
        ) = quantized_kv_cache.update(input_pos, k, v)

        def index(t, input_pos):
            return t[:, :, input_pos, :]

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
        (
            updated_dequantized_k_cache,
            updated_dequantized_v_cache,
        ) = quantized_kv_cache.update(input_pos, k, v)
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

    def test_simple_update_fetch(self):
        self._test_simple_update_fetch()

    def test_simple_update_fetch_use_custom_op(self):
        self._test_simple_update_fetch(use_custom_update_cache_op=True)

    def test_simple_update_fetch_dynamic_shape(self):
        self._test_simple_update_fetch(is_dynamic_shape=True)

    def test_simple_update_fetch_dynamic_shape_use_custom_op(self):
        self._test_simple_update_fetch(
            is_dynamic_shape=True, use_custom_update_cache_op=True
        )

    def _test_update_with_non_float32_activation_dtype(
        self, dtype, use_custom_update_cache_op
    ):
        """QuantizedKVCache stores int8 data internally but always dequantizes
        back to a hardcoded `cache_fp_type` (float32), regardless of the dtype
        of the live k/v activations it's updated with (`cache_fp_type` is not
        parameterized by, and does not track, the model's actual compute
        dtype). When those activations are bfloat16/float16 (the common case
        for on-device LLM inference, which is the whole point of
        ExecuTorch), `QuantizedKVCache.update()` must not crash and must
        return values consistent with the un-quantized reference cache,
        exactly as it already does for float32 activations in
        `_test_simple_update_fetch` above.
        """
        max_batch_size, max_context_len, n_kv_heads, head_dim = 1, 5, 8, 17
        kv_cache = KVCache(
            max_batch_size,
            max_context_len,
            n_kv_heads,
            head_dim,
            False,
            dtype=dtype,
        )
        quantized_kv_cache = QuantizedKVCache.from_float(
            kv_cache,
            QuantizedCacheType.AffineAsymmetric,
            use_custom_update_cache_op,
        )

        input_pos = torch.tensor([0, 1, 2])
        shape = (max_batch_size, n_kv_heads, input_pos.size(0), head_dim)
        k = torch.rand(shape, dtype=dtype)
        v = torch.rand(shape, dtype=dtype)

        updated_dequantized_k_cache, updated_dequantized_v_cache = (
            quantized_kv_cache.update(input_pos, k, v)
        )

        # Reference: un-quantized cache update, same dtype.
        updated_k_cache, updated_v_cache = kv_cache.update(input_pos, k, v)

        def index(t, positions):
            return t[:, :, positions, :]

        torch.testing.assert_close(
            index(updated_k_cache, input_pos).float(),
            index(updated_dequantized_k_cache, input_pos).float(),
            rtol=1e-02,
            atol=1e-02,
        )
        torch.testing.assert_close(
            index(updated_v_cache, input_pos).float(),
            index(updated_dequantized_v_cache, input_pos).float(),
            rtol=1e-02,
            atol=1e-02,
        )

    def test_update_bfloat16_activation(self):
        self._test_update_with_non_float32_activation_dtype(
            torch.bfloat16, use_custom_update_cache_op=False
        )

    def test_update_bfloat16_activation_use_custom_op(self):
        self._test_update_with_non_float32_activation_dtype(
            torch.bfloat16, use_custom_update_cache_op=True
        )

    def test_update_float16_activation(self):
        self._test_update_with_non_float32_activation_dtype(
            torch.float16, use_custom_update_cache_op=False
        )

    def test_update_float16_activation_use_custom_op(self):
        self._test_update_with_non_float32_activation_dtype(
            torch.float16, use_custom_update_cache_op=True
        )
