# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.examples.models.llama.attention import Attention, KVCache, SDPA
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    QuantizedCacheType,
    QuantizedKVCache,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    QuantizedSDPA,
    replace_sdpa_with_custom_op,
    replace_sdpa_with_quantized_sdpa,
    SDPACustom,
)


class MockAttention(Attention):
    """Mock Attention class for testing purposes."""

    def __init__(
        self, dim, head_dim, n_rep, max_context_len=100, enable_dynamic_shape=False
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_rep = n_rep
        self.SDPA = SDPA(dim, head_dim, n_rep, max_context_len)
        self.kv_cache = None

    def forward(self, x, freqs_cos, freqs_sin, **kwargs):
        # Not used in tests
        pass


class QuantizedSDPATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.max_batch_size = 1
        self.max_context_len = 5
        self.n_kv_heads = 4
        self.n_heads = 8
        self.head_dim = 16
        self.dim = self.n_heads * self.head_dim
        self.enable_dynamic_shape = False
        self.dtype = torch.float32

    def _create_test_model(self):
        """Create a simple model with SDPA modules for testing."""
        model = torch.nn.Module()
        attention = MockAttention(
            self.dim, self.head_dim, self.n_heads // self.n_kv_heads
        )
        # Add KVCache to the attention module
        attention.kv_cache = KVCache(
            self.max_batch_size,
            self.max_context_len,
            self.n_kv_heads,
            self.head_dim,
            self.enable_dynamic_shape,
            dtype=self.dtype,
        )
        model.attention = attention
        return model

    def test_replace_sdpa_with_quantized_sdpa(self):
        """Test that replace_sdpa_with_quantized_sdpa correctly transforms SDPA to QuantizedSDPA."""
        # Create a model with SDPA
        model = self._create_test_model()

        # First replace standard SDPA with SDPACustom (required before quantization)
        model = replace_sdpa_with_custom_op(model)
        self.assertIsInstance(model.attention.SDPA, SDPACustom)

        # Replace KVCache with QuantizedKVCache
        model.attention.kv_cache = QuantizedKVCache.from_float(
            model.attention.kv_cache,
            QuantizedCacheType.AffineAsymmetric,
            use_custom_update_cache_op=True,
        )
        self.assertIsInstance(model.attention.kv_cache, QuantizedKVCache)

        # Set return_float_values to False to enable quantized operation
        model.attention.kv_cache.return_float_values = False

        # Apply the transformation
        model = replace_sdpa_with_quantized_sdpa(model)

        # Verify that SDPA has been replaced with QuantizedSDPA
        self.assertIsInstance(model.attention.SDPA, QuantizedSDPA)

        # Verify that the QuantizedSDPA has the correct properties
        self.assertEqual(model.attention.SDPA.dim, self.dim)
        self.assertEqual(model.attention.SDPA.quantized_dtype, torch.int8)
        self.assertEqual(model.attention.SDPA.float_dtype, torch.float32)
        self.assertIs(model.attention.SDPA.kv_cache, model.attention.kv_cache)

    def test_no_replacement_when_no_quantized_kv_cache(self):
        """Test that SDPA is not replaced when there's no QuantizedKVCache."""
        # Create a model with SDPA
        model = self._create_test_model()

        # First replace standard SDPA with SDPACustom
        model = replace_sdpa_with_custom_op(model)
        self.assertIsInstance(model.attention.SDPA, SDPACustom)

        # Apply the transformation without replacing KVCache
        model = replace_sdpa_with_quantized_sdpa(model)

        # Verify that SDPA has NOT been replaced with QuantizedSDPA
        self.assertIsInstance(model.attention.SDPA, SDPACustom)
        self.assertNotIsInstance(model.attention.SDPA, QuantizedSDPA)

    def test_forward_functionality(self):
        """Test that the QuantizedSDPA forward function works correctly."""
        # This test requires the custom ops to be loaded, so we'll check if they're available
        try:
            from executorch.extension.llm.custom_ops import custom_ops  # noqa
        except ImportError:
            self.skipTest(
                "Custom ops not available, skipping forward functionality test"
            )

        # Create a model with SDPA
        model = self._create_test_model()

        # First replace standard SDPA with SDPACustom
        model = replace_sdpa_with_custom_op(model)

        # Replace KVCache with QuantizedKVCache
        model.attention.kv_cache = QuantizedKVCache.from_float(
            model.attention.kv_cache,
            QuantizedCacheType.AffineAsymmetric,
            use_custom_update_cache_op=True,
        )

        # Set return_float_values to False to enable quantized operation
        model.attention.kv_cache.return_float_values = False

        # Save the original SDPACustom for comparison
        # Apply the transformation
        model = replace_sdpa_with_quantized_sdpa(model)

        # Create test inputs
        input_pos = torch.tensor([0], dtype=torch.int64)
        bsz = 1
        seqlen = 1
        q = torch.randn(bsz, self.n_heads, seqlen, self.head_dim, dtype=self.dtype)
        k = torch.randn(bsz, self.n_kv_heads, seqlen, self.head_dim, dtype=self.dtype)
        v = torch.randn(bsz, self.n_kv_heads, seqlen, self.head_dim, dtype=self.dtype)

        # Update the KV cache
        k_quantized, v_quantized = model.attention.kv_cache.update(input_pos, k, v)

        # Run the forward pass with the quantized SDPA
        output = model.attention.SDPA(
            input_pos, q, k_quantized, v_quantized, bsz, seqlen, None
        )

        # Verify the output shape
        self.assertEqual(output.shape, (bsz, seqlen, self.dim))
