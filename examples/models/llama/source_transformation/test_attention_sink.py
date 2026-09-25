# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
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


class RopeWithAttentionSinkWrapTest(unittest.TestCase):
    """get_freqs over a chunk that crosses the top of the ring.

    The cases above all stay below the first wrap, where remapping is the
    identity, so none of them can tell a per-position remap from a remapped
    start plus a contiguous slice. These can.
    """

    SINK_SIZE = 4
    WINDOW_SIZE = 8

    def setUp(self) -> None:
        # Ring top is 20. The table is deliberately longer, so a slice running
        # past the ring still lands on real rows instead of going out of bounds.
        self.params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_context_len=64,
            max_seq_len=self.WINDOW_SIZE,
        )
        self.rope = RopeWithAttentionSink(
            params=self.params,
            window_size=self.WINDOW_SIZE,
            sink_size=self.SINK_SIZE,
        )
        self.ring_top = self.SINK_SIZE + 2 * self.WINDOW_SIZE

    def test_a_chunk_crossing_the_ring_top_wraps(self) -> None:
        """Ground truth: the rows are spelled out, not derived from the code.

        Every other test here checks the implementation against itself -- the
        sweep below uses _remap_input_pos as its own oracle, and the two tests
        after it compare two calls of the same code. None of them would notice
        a wrong modulus or a wrong sink boundary. This one would.
        """
        start, seq_len = 18, 5
        self.assertGreater(start + seq_len, self.ring_top)

        freqs_cos, freqs_sin = self.rope.get_freqs(
            input_pos=torch.tensor([start], dtype=torch.int32), seq_len=seq_len
        )

        # Positions 18..22 with sink_size=4 and a ring of 16: 18 and 19 are
        # still below the ring top, 20 is the first to come back around.
        expected = [18, 19, 4, 5, 6]
        torch.testing.assert_close(freqs_cos, self.rope.freqs_cos[expected])
        torch.testing.assert_close(freqs_sin, self.rope.freqs_sin[expected])

        # A contiguous slice from the remapped start is a different answer, not
        # merely a different spelling of this one.
        sliced = self.rope.freqs_cos.narrow(0, start, seq_len)
        self.assertFalse(torch.allclose(freqs_cos, sliced))

    def test_a_position_gets_the_same_freqs_whatever_chunk_it_lands_in(self) -> None:
        """Chunk size must not change a token's rotation.

        The only test that mixes chunk lengths. The sweep below is seq_len=5
        throughout, and single-token decode is the one shape that is correct
        even without this change, so a disagreement between the two is what
        the previous code produced and what a caller would actually hit.
        """
        # True position 20 is the first position past the ring top. Ask for it
        # as the third entry of a chunk, then as a chunk of its own.
        in_chunk, _ = self.rope.get_freqs(
            input_pos=torch.tensor([18], dtype=torch.int32), seq_len=5
        )
        alone, _ = self.rope.get_freqs(
            input_pos=torch.tensor([20], dtype=torch.int32), seq_len=1
        )

        torch.testing.assert_close(in_chunk[2], alone[0])

    def test_every_chunk_across_four_ring_cycles_gathers_in_bounds(self) -> None:
        """Breadth: every start across four ring cycles, and no index escapes.

        Consistency with _remap_input_pos rather than ground truth, so it
        cannot catch a wrong remap -- it is here for the starts the other
        tests do not name, and for the bound on the gathered indices.
        """
        seq_len = 5
        table_len = self.rope.freqs_cos.shape[0]
        wrapping = 0
        for start in range(4 * self.ring_top):
            with self.subTest(start=start):
                expected = self.rope._remap_input_pos(
                    torch.arange(start, start + seq_len)
                )
                wrapping += int(bool((expected.diff() != 1).any()))
                self.assertGreaterEqual(int(expected.min()), 0)
                self.assertLess(int(expected.max()), table_len)

                freqs_cos, freqs_sin = self.rope.get_freqs(
                    input_pos=torch.tensor([start], dtype=torch.int32), seq_len=seq_len
                )
                torch.testing.assert_close(freqs_cos, self.rope.freqs_cos[expected])
                torch.testing.assert_close(freqs_sin, self.rope.freqs_sin[expected])

        # Guard the fixture: 4 of every 16 starts put a chunk of 5 across the
        # ring top. If a config change made that 0 the sweep would still pass
        # while testing nothing this diff is about.
        self.assertEqual(wrapping, 16)


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
            max_seq_len=self.window_size,
        )
        self.rope = RopeWithAttentionSink(
            params=self.params,
            window_size=self.window_size,
            sink_size=self.sink_size,
        )
        # Total cache size = sink_size + window_size + max_seq_len = 60.
        self.cache_size = self.sink_size + self.window_size + self.params.max_seq_len
        self.kv_cache = KVCacheWithAttentionSink(
            n_heads=self.params.n_heads,
            head_dim=self.params.head_dim,
            enable_dynamic_shape=self.params.enable_dynamic_shape,
            rope=self.rope,
            max_batch_size=self.max_batch_size,
            window_size=self.window_size,
            sink_size=self.sink_size,
            max_context_length=self.params.max_context_len,
            max_seq_len=self.params.max_seq_len,
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
        window_size = 100
        params = ModelArgs(
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_context_len=128,
            max_seq_len=64,
        )
        rope = RopeWithAttentionSink(
            params=params, window_size=window_size, sink_size=0
        )
        cache = KVCacheWithAttentionSink(
            n_heads=params.n_heads,
            head_dim=params.head_dim,
            enable_dynamic_shape=params.enable_dynamic_shape,
            rope=rope,
            max_batch_size=1,
            window_size=window_size,
            sink_size=0,
            max_context_length=params.max_context_len,
            max_seq_len=params.max_seq_len,
            dtype=self.dtype,
        )
        cache_size = window_size + params.max_seq_len
        self.assertEqual(cache_size, 164)
        self.assertEqual(rope.ring_size, cache_size)
        self.assertEqual(cache.max_context_length, cache_size)

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
            model,
            params=args,
            sink_size=sink_size,
            window_size=window_size,
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

    def _feed_in_chunks(self, model, tokens, chunk_size):
        """Feed a fixed token sequence through the model chunk_size at a time.

        Returns one output per chunk. chunk_size=1 is the decode loop; anything
        larger is a chunked prefill, which is the only way to get a multi-token
        chunk at a start position other than 0.
        """
        outputs = []
        with torch.no_grad():
            for pos in range(0, tokens.shape[1], chunk_size):
                result = model(
                    tokens=tokens[:, pos : pos + chunk_size],
                    attn_options={"input_pos": torch.tensor([pos], dtype=torch.long)},
                )
                outputs.append(result[0] if isinstance(result, tuple) else result)
        return outputs

    def test_beyond_context_window_basic(self):
        """Generate tokens well beyond the KV cache size using standard SDPA."""
        sink_size = 4
        window_size = 16
        # KV cache size = sink_size + window_size + max_seq_len = 52
        # max_context_len = 128 (for RoPE table)
        args = self._make_args(max_context_len=128)
        model = self._build_model(args, sink_size, window_size, use_custom_sdpa=False)

        # Generate 80 tokens — beyond the KV cache size of 52
        outputs = self._run_generation(model, args, num_tokens=80)

        self.assertEqual(len(outputs), 77)  # 1 prefill + 76 decode steps
        for out in outputs:
            self.assertTrue(
                torch.isfinite(out).all(), "Output contains non-finite values"
            )

    def test_beyond_max_context_len(self):
        """Generate tokens beyond max_context_len with RoPE position remapping."""
        sink_size = 4
        window_size = 16
        # With max_seq_len omitted, the cache is capped at max_context_len.
        # Generate 100 tokens — well beyond max_context_len
        args = self._make_args(max_context_len=64)
        args.max_seq_len = None
        model = self._build_model(args, sink_size, window_size, use_custom_sdpa=False)
        cache = model.layers[0].attention.kv_cache
        self.assertEqual(cache.max_context_length, 84)

        outputs = self._run_generation(model, args, num_tokens=100)

        self.assertEqual(len(outputs), 97)  # 1 prefill + 96 decode steps
        for out in outputs:
            self.assertTrue(
                torch.isfinite(out).all(),
                "Output contains non-finite values beyond max_context_len",
            )

    def test_chunked_prefill_across_the_ring_wrap(self):
        """Chunked prefill where a chunk spans the ring wrap.

        sink_size=4, window_size=16, and max_seq_len=48, so the ring is slots
        [4, 68). Feeding 40 tokens at a time exceeds the old 2x-window ring and
        crosses the new ring boundary on the second chunk.

        The other beyond-context-window tests decode one token at a time, and a
        chunk of one can never span the wrap however far the position runs, so
        none of them reach this.

        The RopeWithAttentionSinkWrapTest cases call get_freqs directly at
        window_size=8. This is the only one that goes through the model, and
        the only one at window_size=16, so it also covers the mask and the KV
        cache agreeing with the remapped frequencies rather than get_freqs
        alone.
        """
        sink_size = 4
        window_size = 16
        chunk_size = 40
        args = self._make_args(max_context_len=128)
        args.max_seq_len = 48

        torch.manual_seed(0)
        model = self._build_model(args, sink_size, window_size)
        tokens = torch.randint(0, args.vocab_size, (1, 80))

        chunked = self._feed_in_chunks(
            copy.deepcopy(model), tokens, chunk_size=chunk_size
        )
        one_at_a_time = self._feed_in_chunks(copy.deepcopy(model), tokens, chunk_size=1)

        self.assertEqual(len(chunked), 2)
        self.assertEqual(len(one_at_a_time), 80)

        # generate_full_logits is off, so each call returns logits for its last
        # position only. How the input was chunked must not change the result:
        # chunk i ends on the same token as the corresponding decode call.
        for i, out in enumerate(chunked):
            self.assertTrue(torch.isfinite(out).all(), f"chunk {i} is not finite")
            torch.testing.assert_close(
                out,
                one_at_a_time[chunk_size * (i + 1) - 1],
                msg=lambda m, i=i: f"chunk {i}, positions {chunk_size * i}.."
                f"{chunk_size * (i + 1) - 1}, "
                f"disagrees with feeding the same tokens one at a time:\n{m}",
            )

    def test_beyond_context_window_custom_sdpa(self):
        """Generate tokens beyond context window with custom SDPA + custom KV cache."""
        sink_size = 4
        window_size = 16
        args = self._make_args(max_context_len=128)
        model = self._build_model(args, sink_size, window_size, use_custom_sdpa=True)

        # Verify KV caches were replaced with CustomKVCacheWithAttentionSink
        from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
            CustomKVCacheWithAttentionSink,
        )

        found_custom_cache = False
        for m in model.modules():
            if isinstance(m, CustomKVCacheWithAttentionSink):
                found_custom_cache = True
                break
        self.assertTrue(
            found_custom_cache, "Expected CustomKVCacheWithAttentionSink in model"
        )

        # Generate 80 tokens — beyond the KV cache size of 52
        outputs = self._run_generation(model, args, num_tokens=80)

        self.assertEqual(len(outputs), 77)
        for out in outputs:
            self.assertTrue(
                torch.isfinite(out).all(), "Output contains non-finite values"
            )

    def test_sink_zero_custom_sdpa(self):
        """Degenerate case: sink_size=0 with custom SDPA (pure ring buffer)."""
        sink_size = 0
        window_size = 16
        args = self._make_args(max_context_len=128)
        model = self._build_model(args, sink_size, window_size, use_custom_sdpa=True)

        outputs = self._run_generation(model, args, num_tokens=60)

        self.assertEqual(len(outputs), 57)
        for out in outputs:
            self.assertTrue(
                torch.isfinite(out).all(), "Output contains non-finite values"
            )
