# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.llama.model_args import ModelArgs

from executorch.examples.models.llama.source_transformation.attention_sink import (
    CachePositionsManagerWithSink,
    KVCacheWithAttentionSink,
    RopeWithAttentionSink,
)
from parameterized import parameterized


class RopeWithAttentionSinkTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(
            use_kv_cache=True, enable_dynamic_shape=True, max_context_len=256
        )
        self.rope = RopeWithAttentionSink(
            params=self.params,
            window_size=124,
            sink_size=4,
        )

    @parameterized.expand(
        [
            [0, 10],
            [50, 10],
            [200, 10],
            [0, 1],
            [100, 5],
        ]
    )
    def test_get_freqs_passthrough(self, input_pos, seq_len):
        """get_freqs should return frequencies for the exact input position (no shifting)."""
        freqs_cos, freqs_sin = self.rope.get_freqs(
            input_pos=torch.tensor([input_pos], dtype=torch.int32),
            seq_len=seq_len,
        )

        expected_cos = self.rope.freqs_cos.narrow(0, input_pos, seq_len)
        expected_sin = self.rope.freqs_sin.narrow(0, input_pos, seq_len)

        torch.testing.assert_close(freqs_cos, expected_cos)
        torch.testing.assert_close(freqs_sin, expected_sin)


class CachePositionsManagerWithSinkTest(unittest.TestCase):

    def test_sink_indices_fixed(self):
        """Positions < sink_size should map to themselves (fixed slots)."""
        manager = CachePositionsManagerWithSink(cache_size=12, sink_size=4)
        # Fill sink tokens: positions 0,1,2,3
        indices = manager.calculate_positions_and_update_indices(
            torch.tensor([0], dtype=torch.long), seq_len=4
        )
        self.assertEqual(indices.tolist(), [0, 1, 2, 3])

    def test_window_indices_ring_buffer(self):
        """Positions >= sink_size should use ring buffer in [sink_size, cache_size)."""
        manager = CachePositionsManagerWithSink(cache_size=12, sink_size=4)
        # ring_size = 12 - 4 = 8
        # Position 4 -> slot 4, position 5 -> slot 5, etc.
        indices = manager.calculate_positions_and_update_indices(
            torch.tensor([4], dtype=torch.long), seq_len=3
        )
        self.assertEqual(indices.tolist(), [4, 5, 6])

    def test_window_wraps_around(self):
        """Window tokens should wrap around in the ring buffer region."""
        manager = CachePositionsManagerWithSink(cache_size=12, sink_size=4)
        # ring_size = 8, positions 12..14 -> (12-4)%8=0 -> slot 4, slot 5, slot 6
        indices = manager.calculate_positions_and_update_indices(
            torch.tensor([12], dtype=torch.long), seq_len=3
        )
        self.assertEqual(indices.tolist(), [4, 5, 6])

    def test_sink_never_overwritten(self):
        """After wrapping, sink slots (0..sink_size-1) should retain original positions."""
        manager = CachePositionsManagerWithSink(cache_size=12, sink_size=4)
        # Fill sink + some window
        manager.calculate_positions_and_update_indices(
            torch.tensor([0], dtype=torch.long), seq_len=10
        )
        # Wrap around: position 12 maps to slot 4
        manager.calculate_positions_and_update_indices(
            torch.tensor([12], dtype=torch.long), seq_len=3
        )
        # Sink positions should still show 0,1,2,3
        self.assertEqual(manager.cache_positions[:4].tolist(), [0, 1, 2, 3])

    def test_cache_positions_updated(self):
        """cache_positions should track the actual position stored at each slot."""
        manager = CachePositionsManagerWithSink(cache_size=8, sink_size=2)
        # ring_size = 6
        # Fill positions 0..7
        manager.calculate_positions_and_update_indices(
            torch.tensor([0], dtype=torch.long), seq_len=8
        )
        self.assertEqual(manager.cache_positions.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        # Position 8 wraps to slot 2 (sink_size + (8-2)%6 = 2)
        manager.calculate_positions_and_update_indices(
            torch.tensor([8], dtype=torch.long), seq_len=1
        )
        self.assertEqual(manager.cache_positions.tolist(), [0, 1, 8, 3, 4, 5, 6, 7])


class KVCacheWithAttentionSinkTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.max_batch_size = 1
        self.window_size = 28
        self.sink_size = 4
        self.dtype = torch.float32
        self.params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_context_len=256,
        )
        self.rope = RopeWithAttentionSink(
            params=self.params,
            window_size=self.window_size,
            sink_size=self.sink_size,
        )
        # Total cache size = sink_size + window_size * 2 = 4 + 56 = 60
        self.cache_size = self.sink_size + self.window_size * 2
        self.kv_cache = KVCacheWithAttentionSink(
            n_heads=self.params.n_heads,
            head_dim=self.params.head_dim,
            enable_dynamic_shape=self.params.enable_dynamic_shape,
            rope=self.rope,
            max_batch_size=self.max_batch_size,
            window_size=self.window_size,
            sink_size=self.sink_size,
            dtype=self.dtype,
        )

    def _rand_kv(self, seq_len):
        size = (self.max_batch_size, self.params.n_heads, seq_len, self.params.head_dim)
        return torch.rand(*size, dtype=self.dtype), torch.rand(*size, dtype=self.dtype)

    def test_evict_tokens_returns_zero(self):
        """Ring buffer implementation needs no eviction; evict_tokens always returns 0."""
        input_pos = torch.tensor([0], dtype=torch.int32)
        self.assertEqual(self.kv_cache.evict_tokens(input_pos, 1), 0)

        input_pos = torch.tensor([100], dtype=torch.int32)
        self.assertEqual(self.kv_cache.evict_tokens(input_pos, 10), 0)

    def test_update_initial_fill(self):
        """First tokens should fill cache slots sequentially."""
        k, v = self._rand_kv(10)
        input_pos = torch.tensor([0], dtype=torch.long)
        k_out, v_out = self.kv_cache.update(input_pos, k, v)

        # Slots 0..9 should contain our data
        torch.testing.assert_close(k_out[:, :, :10, :], k)
        torch.testing.assert_close(v_out[:, :, :10, :], v)
        # Remaining slots should be zeros
        torch.testing.assert_close(
            k_out[:, :, 10:, :],
            torch.zeros_like(k_out[:, :, 10:, :]),
        )

    def test_sink_tokens_preserved_after_wrap(self):
        """Sink tokens (positions 0..sink_size-1) must never be overwritten."""
        # Fill entire cache
        k_init, v_init = self._rand_kv(self.cache_size)
        input_pos = torch.tensor([0], dtype=torch.long)
        self.kv_cache.update(input_pos, k_init, v_init)

        sink_k = k_init[:, :, : self.sink_size, :].clone()
        sink_v = v_init[:, :, : self.sink_size, :].clone()

        # Write past the cache size — should wrap in window region only
        k_new, v_new = self._rand_kv(5)
        input_pos = torch.tensor([self.cache_size], dtype=torch.long)
        k_out, v_out = self.kv_cache.update(input_pos, k_new, v_new)

        # Sink tokens must be unchanged
        torch.testing.assert_close(k_out[:, :, : self.sink_size, :], sink_k)
        torch.testing.assert_close(v_out[:, :, : self.sink_size, :], sink_v)

    def test_ring_buffer_wrapping(self):
        """Window tokens should wrap correctly in the ring buffer region."""
        ring_size = self.cache_size - self.sink_size  # 56

        # Fill cache initially
        k_init, v_init = self._rand_kv(self.cache_size)
        self.kv_cache.update(torch.tensor([0], dtype=torch.long), k_init, v_init)

        # Write at position that wraps: pos = sink_size + ring_size = 4 + 56 = 60
        # This should map to slot sink_size + (60-4)%56 = 4 + 0 = slot 4
        k_wrap, v_wrap = self._rand_kv(3)
        self.kv_cache.update(
            torch.tensor([self.sink_size + ring_size], dtype=torch.long),
            k_wrap,
            v_wrap,
        )

        # Slots 4,5,6 should now have the new data
        k_out = self.kv_cache.k_cache
        torch.testing.assert_close(
            k_out[:, :, self.sink_size : self.sink_size + 3, :], k_wrap
        )

    def test_sequential_generation(self):
        """Simulate sequential token generation and verify sink protection."""
        # Prefill 10 tokens
        k_prefill, v_prefill = self._rand_kv(10)
        self.kv_cache.update(torch.tensor([0], dtype=torch.long), k_prefill, v_prefill)

        sink_k = k_prefill[:, :, : self.sink_size, :].clone()

        # Generate tokens one by one, well past cache size
        for pos in range(10, self.cache_size + 20):
            k_tok, v_tok = self._rand_kv(1)
            self.kv_cache.update(torch.tensor([pos], dtype=torch.long), k_tok, v_tok)

        # Sink tokens must still be the original ones
        torch.testing.assert_close(
            self.kv_cache.k_cache[:, :, : self.sink_size, :], sink_k
        )

    def test_causal_mask_attends_to_sink(self):
        """The causal mask should always allow attending to sink tokens."""
        # Fill some tokens
        k, v = self._rand_kv(20)
        self.kv_cache.update(torch.tensor([0], dtype=torch.long), k, v)

        # Get mask for position 15
        mask = self.kv_cache.create_causal_mask_for_ring_buffer(start_pos=15, seq_len=1)

        # Sink slots (0..3) should be attended to (mask value = 0, not -inf)
        for i in range(self.sink_size):
            self.assertEqual(
                mask[0, i].item(),
                0.0,
                f"Sink slot {i} should be attendable",
            )

    def test_causal_mask_blocks_future(self):
        """The causal mask should block future (unfilled) positions."""
        # Fill only 5 tokens
        k, v = self._rand_kv(5)
        self.kv_cache.update(torch.tensor([0], dtype=torch.long), k, v)

        mask = self.kv_cache.create_causal_mask_for_ring_buffer(start_pos=4, seq_len=1)

        # Unfilled slots should be masked (-inf)
        for i in range(5, self.cache_size):
            self.assertEqual(
                mask[0, i].item(),
                float("-inf"),
                f"Unfilled slot {i} should be masked",
            )

    @parameterized.expand(
        [
            [0],  # No sink, pure sliding window
        ]
    )
    def test_no_sink_degenerates_to_ring_buffer(self, sink_size):
        """With sink_size=0, behavior should match a plain ring buffer."""
        params = ModelArgs(
            use_kv_cache=True, enable_dynamic_shape=True, max_context_len=256
        )
        rope = RopeWithAttentionSink(
            params=params, window_size=self.window_size, sink_size=0
        )
        cache = KVCacheWithAttentionSink(
            n_heads=params.n_heads,
            head_dim=params.head_dim,
            enable_dynamic_shape=params.enable_dynamic_shape,
            rope=rope,
            max_batch_size=1,
            window_size=self.window_size,
            sink_size=0,
            dtype=self.dtype,
        )
        cache_size = self.window_size * 2  # 56

        # Fill and wrap
        k_init, v_init = self._rand_kv(cache_size)
        cache.update(torch.tensor([0], dtype=torch.long), k_init, v_init)

        k_new, v_new = self._rand_kv(3)
        cache.update(torch.tensor([cache_size], dtype=torch.long), k_new, v_new)

        # Slot 0,1,2 should have new data (no sink protection)
        torch.testing.assert_close(cache.k_cache[:, :, :3, :], k_new)


class AttentionSinkE2ETest(unittest.TestCase):
    """
    End-to-end test: construct a full Transformer with attention sink,
    optionally with custom SDPA + custom KV cache, and generate tokens
    beyond the context window size.
    """

    def _make_args(self, max_context_len=128):
        return ModelArgs(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            hidden_dim=128,
            max_batch_size=1,
            max_seq_len=32,
            max_context_len=max_context_len,
            use_kv_cache=True,
            enable_dynamic_shape=True,
            n_layers=2,
            vocab_size=32,
        )

    def _build_model(self, args, sink_size, window_size, use_custom_sdpa=False):
        from executorch.examples.models.llama.llama_transformer import (
            construct_transformer,
        )
        from executorch.examples.models.llama.source_transformation.attention_sink import (
            enable_attention_sink,
        )

        model = construct_transformer(args)
        model = enable_attention_sink(
            model, params=args, sink_size=sink_size, window_size=window_size
        )

        if use_custom_sdpa:
            from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
                replace_kv_cache_with_custom_kv_cache,
            )
            from executorch.examples.models.llama.source_transformation.sdpa import (
                replace_sdpa_with_custom_op,
            )

            try:
                replace_sdpa_with_custom_op(model)
            except ImportError:
                raise unittest.SkipTest(
                    "Custom SDPA ops not available (missing pybindings)"
                )
            replace_kv_cache_with_custom_kv_cache(model)

        model.eval()
        return model

    def _run_generation(self, model, args, num_tokens):
        """Run prefill + decode for num_tokens total, return all outputs."""
        outputs = []
        with torch.no_grad():
            # Prefill with 4 tokens
            prefill_tokens = torch.randint(0, args.vocab_size, (1, 4))
            result = model(
                tokens=prefill_tokens,
                attn_options={"input_pos": torch.tensor([0], dtype=torch.long)},
            )
            out = result[0] if isinstance(result, tuple) else result
            outputs.append(out)

            # Decode one token at a time
            for pos in range(4, num_tokens):
                token = torch.randint(0, args.vocab_size, (1, 1))
                result = model(
                    tokens=token,
                    attn_options={"input_pos": torch.tensor([pos], dtype=torch.long)},
                )
                out = result[0] if isinstance(result, tuple) else result
                outputs.append(out)

        return outputs

    def test_beyond_context_window_basic(self):
        """Generate tokens well beyond the KV cache size using standard SDPA."""
        sink_size = 4
        window_size = 16
        # KV cache size = sink_size + window_size * 2 = 36
        # max_context_len = 128 (for RoPE table)
        args = self._make_args(max_context_len=128)
        model = self._build_model(args, sink_size, window_size, use_custom_sdpa=False)

        # Generate 80 tokens — well beyond KV cache size of 36
        outputs = self._run_generation(model, args, num_tokens=80)

        self.assertEqual(len(outputs), 77)  # 1 prefill + 76 decode steps
        for out in outputs:
            self.assertTrue(
                torch.isfinite(out).all(), "Output contains non-finite values"
            )
