# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.examples.models.llama.attention import AttentionMHA
from executorch.examples.models.llama.export_llama_lib import _load_llama_model_metadata
from executorch.examples.models.llama.llama_transformer import (
    _is_kv_donor_layer,
    _is_kv_shared_layer,
    construct_transformer,
    Transformer,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope


class TestYOCOHelpers(unittest.TestCase):
    """Tests for YOCO helper functions in llama_transformer.py."""

    def test_is_kv_shared_layer_disabled(self) -> None:
        """When num_kv_shared_layers=0, no layers are shared."""
        for i in range(10):
            self.assertFalse(
                _is_kv_shared_layer(i, n_layers=10, num_kv_shared_layers=0)
            )

    def test_is_kv_shared_layer_enabled(self) -> None:
        """Last num_kv_shared_layers layers should be shared."""
        n_layers = 10
        num_shared = 3
        for i in range(n_layers):
            expected = i >= 7  # layers 7, 8, 9 are shared
            self.assertEqual(
                _is_kv_shared_layer(i, n_layers, num_shared),
                expected,
                f"Layer {i}: expected shared={expected}",
            )

    def test_is_kv_shared_layer_all_shared(self) -> None:
        """All shared: first_shared=0, guard prevents sharing."""
        for i in range(4):
            self.assertFalse(_is_kv_shared_layer(i, n_layers=4, num_kv_shared_layers=4))

    def test_is_kv_donor_layer_disabled(self) -> None:
        """When num_kv_shared_layers=0, no layers are donors."""
        for i in range(10):
            self.assertFalse(_is_kv_donor_layer(i, n_layers=10, num_kv_shared_layers=0))

    def test_is_kv_donor_layer_enabled(self) -> None:
        """Only the layer immediately before the first shared layer is a donor."""
        n_layers = 10
        num_shared = 3
        # Donor should be layer 6 (first_shared - 1 = 10 - 3 - 1 = 6)
        for i in range(n_layers):
            expected = i == 6
            self.assertEqual(
                _is_kv_donor_layer(i, n_layers, num_shared),
                expected,
                f"Layer {i}: expected donor={expected}",
            )

    def test_is_kv_donor_layer_single_shared(self) -> None:
        """With 1 shared layer, the second-to-last layer should be the donor."""
        n_layers = 4
        num_shared = 1
        self.assertFalse(_is_kv_donor_layer(0, n_layers, num_shared))
        self.assertFalse(_is_kv_donor_layer(1, n_layers, num_shared))
        self.assertTrue(_is_kv_donor_layer(2, n_layers, num_shared))
        self.assertFalse(_is_kv_donor_layer(3, n_layers, num_shared))

    def test_donor_and_shared_are_mutually_exclusive(self) -> None:
        """A layer cannot be both a donor and a shared layer."""
        for n_layers in [4, 8, 16]:
            for num_shared in range(1, n_layers):
                for i in range(n_layers):
                    donor = _is_kv_donor_layer(i, n_layers, num_shared)
                    shared = _is_kv_shared_layer(i, n_layers, num_shared)
                    self.assertFalse(
                        donor and shared,
                        f"Layer {i} (n={n_layers}, shared={num_shared})"
                        " is both donor and shared",
                    )


class TestModelArgsYOCO(unittest.TestCase):
    """Tests for num_kv_shared_layers in ModelArgs."""

    def test_default_num_kv_shared_layers(self) -> None:
        args = ModelArgs()
        self.assertEqual(args.num_kv_shared_layers, 0)

    def test_custom_num_kv_shared_layers(self) -> None:
        args = ModelArgs(num_kv_shared_layers=5)
        self.assertEqual(args.num_kv_shared_layers, 5)


class TestAttentionMHAYOCO(unittest.TestCase):
    """Tests for YOCO KV sharing in AttentionMHA."""

    def setUp(self) -> None:
        self.dim = 64
        self.n_heads = 4
        self.n_kv_heads = 4
        self.head_dim = 16
        self.max_context_len = 32
        self.args = ModelArgs(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_batch_size=1,
            max_context_len=self.max_context_len,
            use_kv_cache=True,
            enable_dynamic_shape=True,
        )
        self.rope = Rope(self.args)

    def _make_yoco_args(
        self, n_layers: int = 4, num_kv_shared_layers: int = 2
    ) -> ModelArgs:
        """Helper to create ModelArgs with YOCO configuration."""
        return ModelArgs(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_batch_size=1,
            max_context_len=self.max_context_len,
            use_kv_cache=True,
            enable_dynamic_shape=True,
            n_layers=n_layers,
            num_kv_shared_layers=num_kv_shared_layers,
        )

    def test_shared_layer_has_no_wk_wv(self) -> None:
        """Shared layer should have wk=None, wv=None."""
        # n_layers=4, num_kv_shared_layers=2 -> layers 2,3 are shared
        args = self._make_yoco_args(n_layers=4, num_kv_shared_layers=2)
        rope = Rope(args)

        # Layer 2 is the first shared layer
        attn = AttentionMHA(args, layer_id=2, rope=rope)
        self.assertIsNone(attn.wk, "Shared layer should have wk=None")
        self.assertIsNone(attn.wv, "Shared layer should have wv=None")
        self.assertTrue(attn.is_kv_shared_layer, "Layer 2 should be a shared layer")
        self.assertFalse(
            attn.has_kv_weights, "has_kv_weights should be False for shared layer"
        )

    def test_non_shared_layer_always_has_wk_wv(self) -> None:
        """Non-shared layers (including donor) should always have wk/wv."""
        args = self._make_yoco_args(n_layers=4, num_kv_shared_layers=2)
        rope = Rope(args)

        # Layer 0 is a regular layer
        attn_regular = AttentionMHA(args, layer_id=0, rope=rope)
        self.assertIsNotNone(attn_regular.wk, "Regular layer should have wk")
        self.assertIsNotNone(attn_regular.wv, "Regular layer should have wv")
        self.assertFalse(
            attn_regular.is_kv_shared_layer, "Layer 0 should not be a shared layer"
        )
        self.assertTrue(
            attn_regular.has_kv_weights, "Regular layer should have KV weights"
        )

        # Layer 1 is the donor layer (last layer before shared layers)
        attn_donor = AttentionMHA(args, layer_id=1, rope=rope)
        self.assertIsNotNone(attn_donor.wk, "Donor layer should have wk")
        self.assertIsNotNone(attn_donor.wv, "Donor layer should have wv")
        self.assertFalse(
            attn_donor.is_kv_shared_layer, "Donor layer should not be a shared layer"
        )
        self.assertTrue(attn_donor.has_kv_weights, "Donor layer should have KV weights")

    def test_shared_layer_no_kv_cache(self) -> None:
        """Shared layer should have kv_cache=None."""
        args = self._make_yoco_args(n_layers=4, num_kv_shared_layers=2)
        rope = Rope(args)

        # Layer 2 is a shared layer
        attn = AttentionMHA(args, layer_id=2, rope=rope)
        self.assertIsNone(attn.kv_cache, "Shared layer should have kv_cache=None")

        # Non-shared layer should have kv_cache
        attn_donor = AttentionMHA(args, layer_id=1, rope=rope)
        self.assertIsNotNone(
            attn_donor.kv_cache, "Non-shared layer should have kv_cache"
        )

    def test_forward_without_shared_kv_returns_kv_to_share(self) -> None:
        """Normal forward (no shared_kv) should return kv_to_share in update dict."""
        args = self._make_yoco_args(n_layers=4, num_kv_shared_layers=2)
        rope = Rope(args)
        # Layer 0 is a non-shared (donor-eligible) layer
        attn = AttentionMHA(args, layer_id=0, rope=rope)
        x = torch.randn(1, 1, self.dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        freqs_cos, freqs_sin = rope.get_freqs(input_pos, 1)

        with torch.no_grad():
            output, update = attn(x, freqs_cos, freqs_sin, input_pos=input_pos)

        self.assertEqual(output.shape, (1, 1, self.dim))
        self.assertIsNotNone(update)
        self.assertIn("kv_to_share", update)
        k_shared, v_shared = update["kv_to_share"]
        self.assertEqual(
            k_shared.shape,
            (1, self.n_kv_heads, self.max_context_len, self.head_dim),
        )
        self.assertEqual(
            v_shared.shape,
            (1, self.n_kv_heads, self.max_context_len, self.head_dim),
        )

    def test_forward_with_shared_kv_skips_cache_update(self) -> None:
        """When shared_kv is provided, the layer's own KV cache must not be modified."""
        args = self._make_yoco_args(n_layers=4, num_kv_shared_layers=2)
        rope = Rope(args)
        attn = AttentionMHA(args, layer_id=0, rope=rope)

        # Populate cache at position 0
        x = torch.randn(1, 1, self.dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        freqs_cos, freqs_sin = rope.get_freqs(input_pos, 1)
        with torch.no_grad():
            attn(x, freqs_cos, freqs_sin, input_pos=input_pos)

        cache_k_before = attn.kv_cache.k_cache.clone()

        # Forward with shared_kv at position 1
        shared_k = torch.randn(1, self.n_kv_heads, self.max_context_len, self.head_dim)
        shared_v = torch.randn(1, self.n_kv_heads, self.max_context_len, self.head_dim)
        x2 = torch.randn(1, 1, self.dim)
        input_pos2 = torch.tensor([1], dtype=torch.long)
        freqs_cos2, freqs_sin2 = rope.get_freqs(input_pos2, 1)

        with torch.no_grad():
            output2, update2 = attn(
                x2,
                freqs_cos2,
                freqs_sin2,
                input_pos=input_pos2,
                shared_kv=(shared_k, shared_v),
            )

        self.assertEqual(output2.shape, (1, 1, self.dim))
        self.assertTrue(
            torch.equal(cache_k_before, attn.kv_cache.k_cache),
            "KV cache must not be updated when shared_kv is provided",
        )
        self.assertIsNone(update2, "Shared layers must not return kv_to_share")

    def test_shared_kv_produces_different_output(self) -> None:
        """shared_kv produces different output than normal K/V."""
        args = self._make_yoco_args(n_layers=4, num_kv_shared_layers=2)
        rope = Rope(args)
        attn = AttentionMHA(args, layer_id=0, rope=rope)

        x = torch.randn(1, 1, self.dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        freqs_cos, freqs_sin = rope.get_freqs(input_pos, 1)

        with torch.no_grad():
            out_normal, _ = attn(x, freqs_cos, freqs_sin, input_pos=input_pos)

        # Reset cache
        attn.kv_cache.k_cache.zero_()
        attn.kv_cache.v_cache.zero_()

        shared_k = torch.randn(1, self.n_kv_heads, self.max_context_len, self.head_dim)
        shared_v = torch.randn(1, self.n_kv_heads, self.max_context_len, self.head_dim)

        with torch.no_grad():
            out_shared, _ = attn(
                x,
                freqs_cos,
                freqs_sin,
                input_pos=input_pos,
                shared_kv=(shared_k, shared_v),
            )

        self.assertFalse(
            torch.allclose(out_normal, out_shared, atol=1e-5),
            "Output with shared_kv should differ from normal forward",
        )

    def test_kv_to_share_matches_cache_contents(self) -> None:
        """The returned kv_to_share should reference the full KV cache."""
        args = self._make_yoco_args(n_layers=4, num_kv_shared_layers=2)
        rope = Rope(args)
        attn = AttentionMHA(args, layer_id=0, rope=rope)
        x = torch.randn(1, 1, self.dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        freqs_cos, freqs_sin = rope.get_freqs(input_pos, 1)

        with torch.no_grad():
            _, update = attn(x, freqs_cos, freqs_sin, input_pos=input_pos)

        k_shared, v_shared = update["kv_to_share"]
        self.assertTrue(
            torch.equal(k_shared, attn.kv_cache.k_cache),
            "kv_to_share K should match cache contents",
        )
        self.assertTrue(
            torch.equal(v_shared, attn.kv_cache.v_cache),
            "kv_to_share V should match cache contents",
        )

    def test_kv_to_share_is_same_object_as_cache(self) -> None:
        """kv_to_share should be same tensors as KV cache."""
        args = self._make_yoco_args(n_layers=4, num_kv_shared_layers=2)
        rope = Rope(args)
        attn = AttentionMHA(args, layer_id=0, rope=rope)
        x = torch.randn(1, 1, self.dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        freqs_cos, freqs_sin = rope.get_freqs(input_pos, 1)

        with torch.no_grad():
            _, update = attn(x, freqs_cos, freqs_sin, input_pos=input_pos)

        k_shared, v_shared = update["kv_to_share"]
        self.assertIs(
            k_shared,
            attn.kv_cache.k_cache,
            "kv_to_share K should be the same object as k_cache (not a clone)",
        )
        self.assertIs(
            v_shared,
            attn.kv_cache.v_cache,
            "kv_to_share V should be the same object as v_cache (not a clone)",
        )


class TestTransformerYOCO(unittest.TestCase):
    """Tests for YOCO integration in the Transformer forward loop.

    These tests verify that the YOCO layer loop in Transformer.forward()
    correctly skips shared layers during prefill, passes donor KV to
    shared layers during decode, and cleans up kv_to_share from the
    output update dict.
    """

    def _make_args(self, n_layers: int = 4, num_kv_shared_layers: int = 0) -> ModelArgs:
        return ModelArgs(
            dim=64,
            n_heads=4,
            n_kv_heads=4,
            head_dim=16,
            hidden_dim=128,
            max_batch_size=1,
            max_seq_len=32,
            max_context_len=32,
            use_kv_cache=True,
            enable_dynamic_shape=True,
            n_layers=n_layers,
            num_kv_shared_layers=num_kv_shared_layers,
            vocab_size=100,
        )

    def test_construct_transformer_with_yoco_args(self) -> None:
        """construct_transformer should work with num_kv_shared_layers set."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        self.assertIsInstance(model, Transformer)
        self.assertEqual(len(model.layers), 4)

    def test_transformer_stores_num_kv_shared_layers(self) -> None:
        """Transformer should store num_kv_shared_layers from ModelArgs."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        self.assertEqual(model.num_kv_shared_layers, 2)

    def test_transformer_stores_zero_num_kv_shared_layers(self) -> None:
        """Transformer should store 0 for num_kv_shared_layers when disabled."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=0)
        model = construct_transformer(args)
        self.assertEqual(model.num_kv_shared_layers, 0)

    def test_no_yoco_baseline(self) -> None:
        """With num_kv_shared_layers=0, standard forward works normally."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=0)
        model = construct_transformer(args)
        model.eval()

        tokens = torch.tensor([[1]], dtype=torch.long)
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            result = model(tokens=tokens, attn_options={"input_pos": input_pos})
        # forward may return (logits, attn_options_update) or just logits
        out = result[0] if isinstance(result, tuple) else result
        self.assertEqual(out.shape, (1, args.vocab_size))

    def test_no_yoco_does_not_return_kv_to_share(self) -> None:
        """With num_kv_shared_layers=0, forward should not return kv_to_share."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=0)
        model = construct_transformer(args)
        model.eval()

        tokens = torch.tensor([[1]], dtype=torch.long)
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            result = model(tokens=tokens, attn_options={"input_pos": input_pos})
        # Should be just logits (no tuple with update dict)
        self.assertFalse(
            isinstance(result, tuple),
            "Non-YOCO forward should not return a tuple",
        )

    def test_attention_returns_kv_to_share(self) -> None:
        """Each AttentionMHA layer should return kv_to_share in its update dict."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        model.eval()

        rope = Rope(args)
        x = torch.randn(1, 1, args.dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        freqs_cos, freqs_sin = rope.get_freqs(input_pos, 1)

        with torch.no_grad():
            donor_layer = model.layers[1]
            _, update = donor_layer.attention(
                donor_layer.attention_norm(x),
                freqs_cos,
                freqs_sin,
                input_pos=input_pos,
            )
        self.assertIsNotNone(update)
        self.assertIn("kv_to_share", update)

    def test_shared_layer_accepts_donor_kv(self) -> None:
        """A shared layer should accept shared_kv and produce valid output."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        model.eval()

        rope = Rope(args)
        x = torch.randn(1, 1, args.dim)
        input_pos = torch.tensor([0], dtype=torch.long)
        freqs_cos, freqs_sin = rope.get_freqs(input_pos, 1)

        # Get kv_to_share from donor layer
        with torch.no_grad():
            donor = model.layers[1]
            _, update = donor.attention(
                donor.attention_norm(x),
                freqs_cos,
                freqs_sin,
                input_pos=input_pos,
            )
            shared_kv = update["kv_to_share"]

        # Run shared layer with donor KV
        shared = model.layers[2]
        with torch.no_grad():
            out, update2 = shared.attention(
                shared.attention_norm(x),
                freqs_cos,
                freqs_sin,
                input_pos=input_pos,
                shared_kv=shared_kv,
            )

        self.assertEqual(out.shape, (1, 1, args.dim))
        # Shared layer should not have kv_cache
        self.assertIsNone(shared.attention.kv_cache)
        self.assertIsNone(update2)

    def test_yoco_decode_produces_finite_output(self) -> None:
        """Decode (seqlen=1) with YOCO should produce finite output.

        During decode, the Transformer loop should pass donor KV to
        shared layers and produce valid output.
        """
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        model.eval()

        tokens = torch.tensor([[1]], dtype=torch.long)
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            result = model(tokens=tokens, attn_options={"input_pos": input_pos})

        out = result[0] if isinstance(result, tuple) else result
        self.assertEqual(out.shape, (1, args.vocab_size))
        self.assertTrue(torch.isfinite(out).all(), "Decode output should be finite")

    def test_yoco_prefill_skips_shared_layers(self) -> None:
        """During prefill (seqlen > 1), shared layers should be skipped.

        We verify this by checking that the shared layer's KV cache is
        not populated during prefill.
        """
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        model.eval()

        # Layer 2 is a shared layer — it should have no kv_cache
        shared_layer = model.layers[2]
        self.assertIsNone(shared_layer.attention.kv_cache)

        # Layer 1 is the donor — verify its cache is all zeros before prefill
        donor_layer = model.layers[1]
        cache_before = donor_layer.attention.kv_cache.k_cache.clone()

        # Prefill with seqlen=3
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            result = model(tokens=tokens, attn_options={"input_pos": input_pos})

        out = result[0] if isinstance(result, tuple) else result
        self.assertTrue(torch.isfinite(out).all(), "Prefill output should be finite")

        # Donor layer's cache should have been populated
        self.assertFalse(
            torch.equal(cache_before, donor_layer.attention.kv_cache.k_cache),
            "Donor layer's cache should be populated after prefill",
        )

    def test_yoco_prefill_then_decode(self) -> None:
        """Prefill followed by decode should work end-to-end with YOCO."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        model.eval()

        with torch.no_grad():
            # Prefill: seqlen=3, shared layers are skipped
            result_prefill = model(
                tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
                attn_options={"input_pos": torch.tensor([0], dtype=torch.long)},
            )
            out_prefill = (
                result_prefill[0]
                if isinstance(result_prefill, tuple)
                else result_prefill
            )
            self.assertTrue(torch.isfinite(out_prefill).all())

            # Decode: seqlen=1, shared layers run with donor KV
            result_decode = model(
                tokens=torch.tensor([[4]], dtype=torch.long),
                attn_options={"input_pos": torch.tensor([3], dtype=torch.long)},
            )
            out_decode = (
                result_decode[0] if isinstance(result_decode, tuple) else result_decode
            )
            self.assertEqual(out_decode.shape, (1, args.vocab_size))
            self.assertTrue(torch.isfinite(out_decode).all())

    def test_kv_to_share_not_leaked_in_output(self) -> None:
        """Transformer.forward() should not leak kv_to_share in its return value.

        The YOCO cleanup logic should remove kv_to_share from the update dict
        before returning, and if the dict becomes empty, return just logits.
        """
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        model.eval()

        tokens = torch.tensor([[1]], dtype=torch.long)
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            result = model(tokens=tokens, attn_options={"input_pos": input_pos})

        if isinstance(result, tuple):
            _, update = result
            self.assertNotIn(
                "kv_to_share",
                update,
                "kv_to_share should be removed from output update dict",
            )
        # If result is not a tuple, kv_to_share was cleaned up and the update
        # dict was empty, so forward returned just logits — this is correct.

    def test_yoco_multiple_decode_steps(self) -> None:
        """Multiple decode steps should work correctly with YOCO."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=2)
        model = construct_transformer(args)
        model.eval()

        with torch.no_grad():
            # Prefill
            model(
                tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
                attn_options={"input_pos": torch.tensor([0], dtype=torch.long)},
            )

            # Decode steps
            for pos in range(3, 8):
                result = model(
                    tokens=torch.tensor([[pos + 1]], dtype=torch.long),
                    attn_options={"input_pos": torch.tensor([pos], dtype=torch.long)},
                )
                out = result[0] if isinstance(result, tuple) else result
                self.assertEqual(out.shape, (1, args.vocab_size))
                self.assertTrue(
                    torch.isfinite(out).all(),
                    f"Decode step at pos={pos} should produce finite output",
                )

    def test_prefill_and_decode_produce_finite_output_no_yoco(self) -> None:
        """Prefill then decode should produce finite output without YOCO."""
        args = self._make_args(n_layers=4, num_kv_shared_layers=0)
        model = construct_transformer(args)
        model.eval()

        with torch.no_grad():
            # Prefill
            result_prefill = model(
                tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
                attn_options={"input_pos": torch.tensor([0], dtype=torch.long)},
            )
            out_prefill = (
                result_prefill[0]
                if isinstance(result_prefill, tuple)
                else result_prefill
            )
            self.assertTrue(torch.isfinite(out_prefill).all())

            # Decode
            result_decode = model(
                tokens=torch.tensor([[4]], dtype=torch.long),
                attn_options={"input_pos": torch.tensor([3], dtype=torch.long)},
            )
            out_decode = (
                result_decode[0] if isinstance(result_decode, tuple) else result_decode
            )
            self.assertEqual(out_decode.shape, (1, args.vocab_size))
            self.assertTrue(torch.isfinite(out_decode).all())


class TestLoadLlamaModelMetadataYOCO(unittest.TestCase):
    """Tests for YOCO metadata in _load_llama_model_metadata."""

    def test_metadata_without_yoco(self) -> None:
        """Default (num_kv_shared_layers=0) should not include YOCO metadata."""
        metadata = _load_llama_model_metadata(
            use_kv_cache=True,
            use_sdpa_with_kv_cache=False,
            enable_dynamic_shape=True,
            max_seq_len=32,
            max_context_len=32,
            n_layers=4,
            vocab_size=100,
        )
        self.assertNotIn("get_num_kv_shared_layers", metadata)

    def test_metadata_with_yoco(self) -> None:
        """When num_kv_shared_layers > 0, metadata should include the field."""
        metadata = _load_llama_model_metadata(
            use_kv_cache=True,
            use_sdpa_with_kv_cache=False,
            enable_dynamic_shape=True,
            max_seq_len=32,
            max_context_len=32,
            n_layers=4,
            vocab_size=100,
            num_kv_shared_layers=2,
        )
        self.assertIn("get_num_kv_shared_layers", metadata)
        self.assertEqual(metadata["get_num_kv_shared_layers"], 2)

    def test_metadata_yoco_zero_explicit(self) -> None:
        """num_kv_shared_layers=0 should not include YOCO metadata."""
        metadata = _load_llama_model_metadata(
            use_kv_cache=True,
            use_sdpa_with_kv_cache=False,
            enable_dynamic_shape=True,
            max_seq_len=32,
            max_context_len=32,
            n_layers=4,
            vocab_size=100,
            num_kv_shared_layers=0,
        )
        self.assertNotIn("get_num_kv_shared_layers", metadata)

    def test_metadata_preserves_other_fields(self) -> None:
        """YOCO metadata should not affect other existing metadata fields."""
        metadata = _load_llama_model_metadata(
            use_kv_cache=True,
            use_sdpa_with_kv_cache=False,
            enable_dynamic_shape=True,
            max_seq_len=32,
            max_context_len=64,
            n_layers=8,
            vocab_size=200,
            num_kv_shared_layers=3,
        )
        self.assertEqual(metadata["get_max_seq_len"], 32)
        self.assertEqual(metadata["get_max_context_len"], 64)
        self.assertEqual(metadata["get_n_layers"], 8)
        self.assertEqual(metadata["get_vocab_size"], 200)
        self.assertTrue(metadata["use_kv_cache"])
        self.assertTrue(metadata["enable_dynamic_shape"])
        self.assertEqual(metadata["get_num_kv_shared_layers"], 3)


if __name__ == "__main__":
    unittest.main()
