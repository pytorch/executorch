# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import torch.nn as nn

from executorch.examples.models.llama.attention import (
    Attention,
    AttentionMHA,
    KVCache,
    RingKVCache,
    Rope,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    CustomKVCache,
    CustomRingKVCache,
    QuantizedKVCache,
    QuantizedRingKVCache,
    replace_kv_cache_with_custom_kv_cache,
    replace_kv_cache_with_quantized_kv_cache,
    replace_kv_cache_with_ring_kv_cache,
)


class MockTransformerBlock(nn.Module):
    def __init__(self, attention: Attention):
        super().__init__()
        self.attention = attention


class TestReplaceKVCache(unittest.TestCase):
    def setUp(self):
        # Common parameters for creating attention modules
        self.batch_size = 2
        self.seq_len = 10
        self.dim = 32
        self.n_heads = 4
        self.n_kv_heads = 2
        self.head_dim = 8
        self.max_context_len = 16
        self.enable_dynamic_shape = True

        # Create model args
        self.args = ModelArgs(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_batch_size=self.batch_size,
            max_context_len=self.max_context_len,
            use_kv_cache=True,
            enable_dynamic_shape=self.enable_dynamic_shape,
        )

        # Create a rope instance
        self.rope = Rope(self.args)

    def _create_attention_with_kv_cache(self) -> Attention:
        """Create an attention module with KVCache."""
        return AttentionMHA(self.args, layer_id=0, rope=self.rope)

    def _create_mock_model(self, attention_modules: List[Attention]) -> nn.Module:
        """Create a mock model with transformer blocks containing the given attention modules."""
        model = nn.Module()
        model.layers = nn.ModuleList(
            [MockTransformerBlock(attention) for attention in attention_modules]
        )
        return model

    def test_replace_kv_cache_with_ring_kv_cache(self):
        """Test replacing KVCache with RingKVCache."""
        # Create a model with KVCache
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])

        # Verify that the model has KVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, KVCache)
        self.assertNotIsInstance(model.layers[0].attention.kv_cache, RingKVCache)

        # Replace KVCache with RingKVCache
        layer_sizes = [8]  # Sliding window size for each layer
        replace_kv_cache_with_ring_kv_cache(model, layer_sizes)

        # Verify that KVCache has been replaced with RingKVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, RingKVCache)

        # Verify that the sliding window size is set correctly
        self.assertEqual(model.layers[0].attention.kv_cache.window_size, layer_sizes[0])

    def test_replace_custom_kv_cache_with_custom_ring_kv_cache(self):
        """Test replacing CustomKVCache with CustomRingKVCache."""
        # Create a model with KVCache
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])

        # Replace KVCache with CustomKVCache
        replace_kv_cache_with_custom_kv_cache(model)

        # Verify that the model has CustomKVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, CustomKVCache)
        self.assertNotIsInstance(model.layers[0].attention.kv_cache, CustomRingKVCache)

        # Replace CustomKVCache with CustomRingKVCache
        layer_sizes = [8]  # Sliding window size for each layer
        replace_kv_cache_with_ring_kv_cache(model, layer_sizes)

        # Verify that CustomKVCache has been replaced with CustomRingKVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, CustomRingKVCache)

    def test_replace_quantized_kv_cache_with_quantized_ring_kv_cache(self):
        """Test replacing QuantizedKVCache with QuantizedRingKVCache."""
        # Create a model with KVCache
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])

        # Replace KVCache with QuantizedKVCache
        replace_kv_cache_with_quantized_kv_cache(model)

        # Verify that the model has QuantizedKVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, QuantizedKVCache)
        self.assertNotIsInstance(
            model.layers[0].attention.kv_cache, QuantizedRingKVCache
        )

        # Replace QuantizedKVCache with QuantizedRingKVCache
        layer_sizes = [8]  # Sliding window size for each layer
        replace_kv_cache_with_ring_kv_cache(model, layer_sizes)

        # Verify that QuantizedKVCache has been replaced with QuantizedRingKVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, QuantizedRingKVCache)

    def test_multiple_layers_with_different_window_sizes(self):
        """Test replacing KV caches in multiple layers with different window sizes."""
        # Create a model with multiple layers
        attention1 = self._create_attention_with_kv_cache()
        attention2 = self._create_attention_with_kv_cache()
        attention3 = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention1, attention2, attention3])

        # Replace KVCache with RingKVCache with different window sizes
        layer_sizes = [4, 8, 16]  # Different sliding window sizes for each layer
        replace_kv_cache_with_ring_kv_cache(model, layer_sizes)

        # Verify that each layer has the correct window size
        self.assertIsInstance(model.layers[0].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[0].attention.kv_cache.window_size, layer_sizes[0])

        self.assertIsInstance(model.layers[1].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[1].attention.kv_cache.window_size, layer_sizes[1])

        self.assertIsInstance(model.layers[2].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[2].attention.kv_cache.window_size, layer_sizes[2])
