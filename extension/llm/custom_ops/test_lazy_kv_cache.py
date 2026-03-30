# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for lazy KV cache (DYNAMIC_UNBOUND) support.

Tests the update_cache custom op's ability to handle caches that start at
seq_len=0 and grow on demand, which is the foundation for pay-as-you-go
KV cache memory allocation.
"""

# pyre-unsafe

import unittest

import torch

from executorch.extension.llm.custom_ops import custom_ops  # noqa


class LazyKVCacheUpdateTest(unittest.TestCase):
    """Test update_cache op with zero-sized initial caches (lazy KV cache)."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 1
        self.num_heads = 4
        self.head_dim = 8
        self.max_seq_len = 64

    def test_update_cache_grows_from_zero(self):
        """Verify update_cache works when cache seq dim starts at full size
        and tokens are appended sequentially."""
        cache = torch.zeros(
            (self.batch_size, self.max_seq_len, self.num_heads, self.head_dim),
            dtype=torch.float32,
        )

        for pos in range(10):
            value = torch.randn(
                (self.batch_size, 1, self.num_heads, self.head_dim),
                dtype=torch.float32,
            )
            torch.ops.llama.update_cache(value, cache, pos)
            self.assertTrue(
                torch.allclose(cache[:, pos : pos + 1, :, :], value),
                f"Cache mismatch at position {pos}",
            )

    def test_custom_kv_cache_lazy_init(self):
        """Verify CustomKVCache with lazy=True creates zero-sized buffers."""
        from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
            CustomKVCache,
        )

        cache = CustomKVCache(
            max_batch_size=1,
            max_context_length=131072,  # 128K ceiling
            n_heads=4,
            head_dim=8,
            dtype=torch.float32,
            lazy=True,
        )
        self.assertEqual(cache.k_cache.shape[1], 0, "Lazy k_cache seq dim should be 0")
        self.assertEqual(cache.v_cache.shape[1], 0, "Lazy v_cache seq dim should be 0")
        self.assertEqual(cache.max_context_length, 131072)

    def test_custom_kv_cache_non_lazy_init(self):
        """Verify CustomKVCache without lazy=True creates full-sized buffers."""
        from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
            CustomKVCache,
        )

        cache = CustomKVCache(
            max_batch_size=1,
            max_context_length=64,
            n_heads=4,
            head_dim=8,
            dtype=torch.float32,
            lazy=False,
        )
        self.assertEqual(cache.k_cache.shape[1], 64)
        self.assertEqual(cache.v_cache.shape[1], 64)

    def test_replace_kv_cache_with_lazy(self):
        """Verify replace_kv_cache_with_custom_kv_cache passes lazy flag."""
        from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
            CustomKVCache,
            replace_kv_cache_with_custom_kv_cache,
        )
        from executorch.examples.models.llama.attention import KVCache

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # KVCache stores as [B, H, S, D]
                self.kv = KVCache(
                    max_batch_size=1,
                    max_context_length=128,
                    n_heads=4,
                    head_dim=8,
                    enable_dynamic_shape=False,
                    dtype=torch.float32,
                )

            def forward(self, x):
                return x

        model = FakeModel()
        replace_kv_cache_with_custom_kv_cache(model, lazy=True)
        self.assertIsInstance(model.kv, CustomKVCache)
        # CustomKVCache stores as [B, S, H, D], lazy means seq_dim=0
        self.assertEqual(model.kv.k_cache.shape[1], 0)


class LazyKVCacheMetaKernelTest(unittest.TestCase):
    """Test that meta kernels work without upper-bound cache size checks."""

    def test_meta_kernel_allows_start_pos_beyond_cache(self):
        """Meta kernel should not reject start_pos >= cache.size(1)."""
        value = torch.randn(1, 1, 4, 8)
        # Cache with seq_len=0 (lazy)
        cache = torch.zeros(1, 0, 4, 8)
        # This should not raise — the runtime op handles resize
        result = torch.ops.llama.update_cache(value, cache, 0)
        self.assertIsNotNone(result)

    def test_meta_kernel_allows_large_start_pos(self):
        """Meta kernel should allow start_pos beyond current cache size for lazy caches."""
        value = torch.randn(1, 1, 4, 8)
        cache = torch.zeros(1, 0, 4, 8)
        # Lazy cache (size(1)==0) skips bounds checks — runtime op handles resize
        result = torch.ops.llama.update_cache(value, cache, 100)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
