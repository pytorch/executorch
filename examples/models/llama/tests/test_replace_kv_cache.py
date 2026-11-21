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

    def test_replace_static_quantized_kv_cache_with_int16_storage(self):
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])

        replace_kv_cache_with_static_quantized_kv_cache(
            model,
            scale=1.0 / 32767.0,
            use_custom_update_cache_op=False,
            use_per_channel=False,
            quantized_cache_dtype=torch.int16,
        )

        cache = model.layers[0].attention.kv_cache
        self.assertIsInstance(cache, StaticQuantizedKVCache)
        self.assertEqual(cache.k_cache.dtype, torch.int16)
        self.assertEqual(cache.v_cache.dtype, torch.int16)
        self.assertTrue(cache.return_float_values)
        self.assertFalse(cache.use_per_channel)

    def test_calibrate_static_quantized_int16_kv_cache(self):
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])
        replace_kv_cache_with_static_quantized_kv_cache(
            model,
            scale=1.0 / 32767.0,
            use_custom_update_cache_op=False,
            use_per_channel=False,
            quantized_cache_dtype=torch.int16,
        )
        cache = model.layers[0].attention.kv_cache
        cache.enable_calibration()

        input_pos = torch.tensor([0, 1])
        shape = (self.batch_size, self.n_kv_heads, 2, self.head_dim)
        k_val = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(
            shape
        )
        v_val = -2.0 * k_val
        cache.update(input_pos, k_val, v_val)
        cache.finalize_calibration()

        self.assertAlmostEqual(
            cache.k_cache_scales.item(),
            k_val.abs().amax().item() / 32767.0,
            places=8,
        )
        self.assertAlmostEqual(
            cache.v_cache_scales.item(),
            v_val.abs().amax().item() / 32767.0,
            places=8,
        )

    def test_static_quantized_int16_cache_round_trip_after_calibration(self):
        for per_channel in (False, True):
            with self.subTest(per_channel=per_channel):
                cache = StaticQuantizedKVCache(
                    1,
                    4,
                    1,
                    4,
                    use_custom_update_cache_op=False,
                    return_float_values=False,
                    use_per_channel=per_channel,
                    quantized_cache_dtype=torch.int16,
                )
                k_val = torch.tensor([[[[-4.0, -0.5, 0.5, 4.0]]]])
                v_val = 0.5 * k_val
                cache.enable_calibration()
                cache.update(torch.tensor([0]), k_val, v_val)
                cache.finalize_calibration()

                k_val = torch.cat((k_val, 0.5 * k_val), dim=2)
                v_val = torch.cat((v_val, 0.5 * v_val), dim=2)
                k_out, v_out = cache.update(torch.tensor([1, 2]), k_val, v_val)

                self.assertEqual(k_out.dtype, torch.int16)
                self.assertEqual(v_out.dtype, torch.int16)
                for stored, scales, expected in (
                    (cache.k_cache, cache.k_cache_scales, k_val),
                    (cache.v_cache, cache.v_cache_scales, v_val),
                ):
                    values = stored[:, 1:3]
                    self.assertEqual(values.abs().max().item(), 32767)
                    torch.testing.assert_close(
                        values.to(torch.float32) * scales,
                        expected.transpose(1, 2),
                        rtol=0,
                        atol=0.5 * scales.max().item() + torch.finfo(torch.float32).eps,
                    )

    def test_static_quantized_int16_custom_update_matches_tensor_update(self):
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

        for per_channel in (False, True):
            with self.subTest(per_channel=per_channel):
                actual = StaticQuantizedKVCache(
                    1,
                    4,
                    2,
                    4,
                    return_float_values=False,
                    use_per_channel=per_channel,
                    quantized_cache_dtype=torch.int16,
                )
                expected = StaticQuantizedKVCache(
                    1,
                    4,
                    2,
                    4,
                    use_custom_update_cache_op=False,
                    return_float_values=False,
                    use_per_channel=per_channel,
                    quantized_cache_dtype=torch.int16,
                )
                self.assertTrue(actual.use_custom_update_cache_op)
                k_val = torch.linspace(-4.0, 4.0, 16).reshape(1, 2, 2, 4)
                v_val = 0.5 * k_val
                for cache in (actual, expected):
                    cache.enable_calibration()
                    cache.update(torch.tensor([0, 1]), k_val, v_val)
                    cache.finalize_calibration()
                for pos, keys, values in (
                    (torch.tensor([1, 2]), k_val, v_val),
                    (torch.tensor([3]), 0.5 * k_val[:, :, :1], v_val[:, :, :1]),
                ):
                    actual_out = actual.update(pos, keys, values)
                    expected_out = expected.update(pos, keys, values)
                    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual.k_cache, expected.k_cache, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual.v_cache, expected.v_cache, rtol=0, atol=0
                    )
                self.assertGreater(actual.k_cache.abs().max().item(), 127)

    def test_replace_static_quantized_cache_logs_dtype_and_initial_scale(self):
        for scale in (None, 0.25):
            for per_channel in (False, True):
                with self.subTest(scale=scale, per_channel=per_channel):
                    attention = self._create_attention_with_kv_cache()
                    model = self._create_mock_model([attention])
                    with self.assertLogs(level="DEBUG") as logs:
                        replace_kv_cache_with_static_quantized_kv_cache(
                            model,
                            scale=scale,
                            use_custom_update_cache_op=False,
                            use_per_channel=per_channel,
                            quantized_cache_dtype=torch.int16,
                        )
                    messages = [
                        record
                        for record in logs.records
                        if "Replacing KVCache" in record.msg
                    ]
                    self.assertEqual(len(messages), 1)
                    self.assertEqual(messages[0].levelname, "DEBUG")
                    message = messages[0].getMessage()
                    self.assertIn("quantized_cache_dtype=torch.int16", message)
                    self.assertIn(f"use_per_channel={per_channel}", message)
                    expected_scale = 1.0 / 32767 if scale is None else scale
                    self.assertIn(f"initial_scale={expected_scale}", message)

    def test_replace_multiple_static_quantized_caches_does_not_log_at_info(self):
        model = self._create_mock_model(
            [self._create_attention_with_kv_cache() for _ in range(2)]
        )
        with self.assertNoLogs(level="INFO"):
            replace_kv_cache_with_static_quantized_kv_cache(
                model, use_custom_update_cache_op=False
            )
        for layer in model.layers:
            self.assertIsInstance(layer.attention.kv_cache, StaticQuantizedKVCache)

    def test_replace_static_quantized_cache_rejects_dtype_before_logging(self):
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])
        original_cache = attention.kv_cache
        with self.assertNoLogs(level="INFO"):
            with self.assertRaisesRegex(ValueError, "storage must use INT8 or INT16"):
                replace_kv_cache_with_static_quantized_kv_cache(
                    model,
                    use_custom_update_cache_op=False,
                    quantized_cache_dtype=torch.uint8,
                )
        self.assertIs(model.layers[0].attention.kv_cache, original_cache)

    def test_static_quantized_cache_default_scale_matches_storage_dtype(self):
        for cache_dtype in (torch.int8, torch.int16):
            for entry_point in ("constructor", "from_float", "replace"):
                with self.subTest(cache_dtype=cache_dtype, entry_point=entry_point):
                    if entry_point == "constructor":
                        cache = StaticQuantizedKVCache(
                            1,
                            4,
                            1,
                            4,
                            quantized_cache_dtype=cache_dtype,
                            use_custom_update_cache_op=False,
                        )
                    elif entry_point == "from_float":
                        cache = StaticQuantizedKVCache.from_float(
                            KVCache(1, 4, 1, 4, False),
                            quantized_cache_dtype=cache_dtype,
                            use_custom_update_cache_op=False,
                        )
                    else:
                        attention = self._create_attention_with_kv_cache()
                        replace_kv_cache_with_static_quantized_kv_cache(
                            self._create_mock_model([attention]),
                            quantized_cache_dtype=cache_dtype,
                            use_custom_update_cache_op=False,
                        )
                        cache = attention.kv_cache
                    expected_scale = 1.0 / torch.iinfo(cache_dtype).max
                    self.assertEqual(cache.k_cache_scale, expected_scale)
                    self.assertEqual(cache.v_cache_scale, expected_scale)
                    for scales in (cache.k_cache_scales, cache.v_cache_scales):
                        torch.testing.assert_close(
                            scales, torch.full_like(scales, expected_scale)
                        )

    def test_static_quantized_cache_rejects_unsupported_storage_dtype(self):
        for cache_dtype in (torch.uint8, torch.float16):
            with self.subTest(cache_dtype=cache_dtype):
                with self.assertRaisesRegex(ValueError, "INT8 or INT16"):
                    StaticQuantizedKVCache(
                        1,
                        4,
                        1,
                        4,
                        quantized_cache_dtype=cache_dtype,
                        use_custom_update_cache_op=False,
                    )

    def test_static_quantized_cache_uses_initial_scale(self):
        for per_channel in (False, True):
            with self.subTest(per_channel=per_channel):
                cache = StaticQuantizedKVCache(
                    1,
                    4,
                    2,
                    4,
                    scale=0.25,
                    use_custom_update_cache_op=False,
                    return_float_values=False,
                    use_per_channel=per_channel,
                )
                for scales in (cache.k_cache_scales, cache.v_cache_scales):
                    torch.testing.assert_close(scales, torch.full_like(scales, 0.25))

                k_val = torch.full((1, 2, 1, 4), 0.5)
                v_val = torch.full_like(k_val, -1.0)
                cache.update(torch.tensor([0]), k_val, v_val)
                k_out, v_out = cache.update(
                    torch.tensor([1]), 2.0 * k_val, -1.5 * v_val
                )

                for output, expected in (
                    (k_out[:, :, 0], 2),
                    (v_out[:, :, 0], -4),
                    (k_out[:, :, 1], 4),
                    (v_out[:, :, 1], 6),
                ):
                    self.assertEqual(output.dtype, torch.int8)
                    torch.testing.assert_close(
                        output, torch.full_like(output, expected)
                    )

    def test_static_quantized_cache_scalar_calibration(self):
        attention = self._create_attention_with_kv_cache()
        model = self._create_mock_model([attention])
        replace_kv_cache_with_static_quantized_kv_cache(
            model,
            use_custom_update_cache_op=False,
            use_per_channel=False,
        )
        cache = model.layers[0].attention.kv_cache
        cache.enable_calibration()
        k_val = torch.randn(self.batch_size, self.n_kv_heads, 1, self.head_dim)
        v_val = torch.randn_like(k_val)
        cache.update(torch.tensor([0]), k_val, v_val)
        cache.finalize_calibration()

        self.assertEqual(cache.k_cache_scales.shape, (1,))
        self.assertEqual(cache.v_cache_scales.shape, (1,))
        torch.testing.assert_close(
            cache.k_cache_scales, k_val.abs().amax().reshape(1) / 127.0
        )
        torch.testing.assert_close(
            cache.v_cache_scales, v_val.abs().amax().reshape(1) / 127.0
        )

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

    def test_replace_static_quantized_cache_preserves_sdpa(self):
        for cache_dtype in (torch.int8, torch.int16):
            with self.subTest(cache_dtype=cache_dtype):
                attention = self._create_attention_with_kv_cache()
                model = self._create_mock_model([attention])
                original_sdpa = attention.SDPA
                replace_kv_cache_with_static_quantized_kv_cache(
                    model,
                    use_custom_update_cache_op=False,
                    quantized_cache_dtype=cache_dtype,
                )
                self.assertIs(attention.SDPA, original_sdpa)
                self.assertTrue(attention.kv_cache.return_float_values)
                generator = torch.Generator().manual_seed(31)
                q = torch.randn(2, 4, 1, 8, generator=generator)
                k = torch.randn(2, 2, 1, 8, generator=generator) * 0.2
                v = torch.randn(2, 2, 1, 8, generator=generator) * 0.2
                history_k = torch.zeros(2, 2, self.max_context_len, 8)
                history_v = torch.zeros_like(history_k)
                for position in range(2):
                    input_pos = torch.tensor([position])
                    k_out, v_out = attention.kv_cache.update(input_pos, k, v)
                    self.assertEqual(k_out.dtype, torch.float32)
                    self.assertEqual(v_out.dtype, torch.float32)
                    history_k[:, :, position : position + 1] = k
                    history_v[:, :, position : position + 1] = v
                    mask = (torch.arange(self.max_context_len) <= position).reshape(
                        1, 1, 1, self.max_context_len
                    )
                    expected = original_sdpa(
                        input_pos, q, history_k, history_v, 2, 1, mask
                    )
                    actual = attention.SDPA(input_pos, q, k_out, v_out, 2, 1, mask)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0.01)
                    k = k * 0.5
                    v = v * -0.5

    def test_static_quantized_kv_cache_rejects_specialized_cache(self):
        attention = self._create_attention_with_kv_cache()
        attention.kv_cache = RingKVCache(
            self.batch_size,
            self.max_context_len,
            self.n_kv_heads,
            self.head_dim,
            self.enable_dynamic_shape,
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
        replace_kv_cache_with_ring_kv_cache(model, layer_sizes)

        # Verify that each layer has the correct window size
        self.assertIsInstance(model.layers[0].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[0].attention.kv_cache.window_size, layer_sizes[0])

        self.assertIsInstance(model.layers[1].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[1].attention.kv_cache.window_size, layer_sizes[1])

        self.assertIsInstance(model.layers[2].attention.kv_cache, RingKVCache)
        self.assertEqual(model.layers[2].attention.kv_cache.window_size, layer_sizes[2])
