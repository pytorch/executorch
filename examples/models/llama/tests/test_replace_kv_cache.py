# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import torch
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
    replace_kv_cache_with_static_quantized_kv_cache,
    StaticQuantizedKVCache,
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
        replace_kv_cache_with_ring_kv_cache(model, layer_sizes, max_seq_len=4)

        # Verify that KVCache has been replaced with RingKVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, RingKVCache)

        # Verify that the sliding window size is set correctly
        self.assertEqual(model.layers[0].attention.kv_cache.window_size, layer_sizes[0])
        self.assertEqual(model.layers[0].attention.kv_cache.k_cache.size(2), 12)

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
        replace_kv_cache_with_ring_kv_cache(model, layer_sizes, max_seq_len=4)

        # Verify that CustomKVCache has been replaced with CustomRingKVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, CustomRingKVCache)
        self.assertEqual(model.layers[0].attention.kv_cache.k_cache.size(1), 12)

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
        replace_kv_cache_with_ring_kv_cache(model, layer_sizes, max_seq_len=4)

        # Verify that QuantizedKVCache has been replaced with QuantizedRingKVCache
        self.assertIsInstance(model.layers[0].attention.kv_cache, QuantizedRingKVCache)
        self.assertEqual(model.layers[0].attention.kv_cache.k_cache.size(1), 12)

    def test_replace_static_quantized_kv_cache(self):
        """Test replacing KVCache with static-qparams int8 KV storage."""
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])

        replace_kv_cache_with_static_quantized_kv_cache(
            model, scale=0.25, use_custom_update_cache_op=False
        )

        cache = model.layers[0].attention.kv_cache
        self.assertIsInstance(cache, StaticQuantizedKVCache)
        self.assertFalse(cache.use_custom_update_cache_op)
        self.assertEqual(cache.k_cache.dtype, cache.quantized_cache_dtype)
        self.assertEqual(cache.k_cache_scales.shape[-1], self.head_dim)
        self.assertIsNone(cache.k_calibration_cache)
        self.assertIsNone(cache.v_calibration_cache)

    def test_calibrate_static_quantized_kv_cache(self):
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])
        replace_kv_cache_with_static_quantized_kv_cache(
            model, scale=0.25, use_custom_update_cache_op=False
        )
        cache = model.layers[0].attention.kv_cache
        self.assertIsNone(cache.k_calibration_cache)
        self.assertIsNone(cache.v_calibration_cache)
        cache.enable_calibration()
        self.assertEqual(cache.k_calibration_cache.shape, cache.k_cache.shape)
        self.assertEqual(cache.v_calibration_cache.shape, cache.v_cache.shape)

        input_pos = torch.tensor([0, 1])
        shape = (self.batch_size, self.n_kv_heads, 2, self.head_dim)
        k_val = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(
            shape
        )
        v_val = -2.0 * k_val
        k_out, v_out = cache.update(input_pos, k_val, v_val)

        torch.testing.assert_close(k_out[:, :, input_pos], k_val)
        torch.testing.assert_close(v_out[:, :, input_pos], v_val)
        expected_k_scales = k_val.abs().amax(dim=(0, 1, 2)) / 127.0
        expected_v_scales = v_val.abs().amax(dim=(0, 1, 2)) / 127.0

        cache.finalize_calibration()

        self.assertFalse(cache.calibration_enabled)
        self.assertIsNone(cache.k_calibration_cache)
        self.assertIsNone(cache.v_calibration_cache)
        torch.testing.assert_close(cache.k_cache_scales.flatten(), expected_k_scales)
        torch.testing.assert_close(cache.v_cache_scales.flatten(), expected_v_scales)
        self.assertEqual(torch.count_nonzero(cache.k_cache), 0)
        self.assertEqual(torch.count_nonzero(cache.v_cache), 0)

    def test_static_quantized_kv_cache_warns_for_zero_channel(self):
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])
        replace_kv_cache_with_static_quantized_kv_cache(
            model, scale=0.25, use_custom_update_cache_op=False
        )
        cache = model.layers[0].attention.kv_cache
        cache.enable_calibration()

        input_pos = torch.tensor([0])
        shape = (self.batch_size, self.n_kv_heads, 1, self.head_dim)
        k_val = torch.ones(shape, dtype=torch.float32)
        v_val = torch.ones(shape, dtype=torch.float32)
        k_val[..., 0] = 0

        cache.update(input_pos, k_val, v_val)
        with self.assertLogs(level="WARNING") as logs:
            cache.finalize_calibration()

        self.assertIn("all-zero K/V channel", logs.output[0])
        self.assertEqual(
            cache.k_cache_scales[..., 0].item(), torch.finfo(torch.float32).tiny
        )

    def test_static_quantized_kv_cache_preserves_small_scales(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                attention = self._create_attention_with_kv_cache()
                model = self._create_mock_model([attention])
                replace_kv_cache_with_static_quantized_kv_cache(
                    model, scale=0.25, use_custom_update_cache_op=False
                )
                model.to(dtype=dtype)

                cache = model.layers[0].attention.kv_cache
                cache.enable_calibration()
                input_pos = torch.tensor([0])
                shape = (self.batch_size, self.n_kv_heads, 1, self.head_dim)
                k_val = torch.full(shape, 0.0625, dtype=dtype)
                v_val = torch.full(shape, -0.03125, dtype=dtype)
                cache.update(input_pos, k_val, v_val)
                cache.finalize_calibration()

                expected_k_scale = (k_val.abs().amax() / 127.0).to(
                    cache.k_cache_scales.dtype
                )
                expected_v_scale = (v_val.abs().amax() / 127.0).to(
                    cache.v_cache_scales.dtype
                )
                torch.testing.assert_close(
                    cache.k_cache_scales,
                    torch.full_like(cache.k_cache_scales, expected_k_scale),
                )
                torch.testing.assert_close(
                    cache.v_cache_scales,
                    torch.full_like(cache.v_cache_scales, expected_v_scale),
                )
                self.assertLess(
                    cache.k_cache_scales.max().item(), torch.finfo(dtype).eps
                )

    def test_static_quantized_kv_cache_preserves_model_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            for cast_before_replacement in (True, False):
                with self.subTest(
                    dtype=dtype, cast_before_replacement=cast_before_replacement
                ):
                    attention = self._create_attention_with_kv_cache()
                    model = self._create_mock_model([attention])
                    if cast_before_replacement:
                        model.to(dtype=dtype)
                    replace_kv_cache_with_static_quantized_kv_cache(
                        model, scale=0.25, use_custom_update_cache_op=False
                    )
                    if not cast_before_replacement:
                        model.to(dtype=dtype)

                    cache = model.layers[0].attention.kv_cache
                    input_pos = torch.tensor([0, 1])
                    shape = (self.batch_size, self.n_kv_heads, 2, self.head_dim)
                    k_val = torch.randn(shape, dtype=dtype)
                    v_val = torch.randn(shape, dtype=dtype)
                    k_out, v_out = cache.update(input_pos, k_val, v_val)

                    self.assertEqual(k_out.dtype, dtype)
                    self.assertEqual(v_out.dtype, dtype)

    def test_static_quantized_kv_cache_rejects_specialized_cache(self):
        attention = self._create_attention_with_kv_cache()
        attention.kv_cache = RingKVCache(
            self.batch_size,
            self.max_context_len,
            self.n_kv_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            window_size=self.max_context_len,
            max_seq_len=self.max_context_len,
        )
        model = self._create_mock_model([attention])

        with self.assertRaisesRegex(ValueError, "RingKVCache"):
            replace_kv_cache_with_static_quantized_kv_cache(
                model, use_custom_update_cache_op=False
            )

    def test_multiple_layers_with_different_window_sizes(self):
        """Test replacing KV caches in multiple layers with different window sizes."""
        # Create a model with multiple layers
        attention1 = self._create_attention_with_kv_cache()
        attention2 = self._create_attention_with_kv_cache()
        attention3 = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention1, attention2, attention3])

        # Replace KVCache with RingKVCache with different window sizes
        layer_sizes = [4, 8, 16]  # Different sliding window sizes for each layer
        replace_kv_cache_with_ring_kv_cache(
            model, layer_sizes, max_seq_len=self.max_context_len
        )

        # Verify that each layer has the correct window size
        self.assertIsInstance(model.layers[0].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[0].attention.kv_cache.window_size, layer_sizes[0])

        self.assertIsInstance(model.layers[1].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[1].attention.kv_cache.window_size, layer_sizes[1])

        self.assertIsInstance(model.layers[2].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[2].attention.kv_cache.window_size, layer_sizes[2])
