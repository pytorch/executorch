# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from executorch.extension.llm.cache.reference_cache import (
    attend,
    AttendSpec,
    BatchedSequenceReferenceCache,
    CacheConfig,
    CacheSizing,
    CellReferenceCache,
    flatten_step,
    LayerKind,
    LayerPolicy,
    MaskKind,
    MAX_SEQS,
    SequenceReferenceCache,
)
from executorch.extension.llm.cache.update_and_attend import REGISTRY, update_and_attend


class TinyAttentionModel(torch.nn.Module):
    # A minimal multi-layer attention stack that calls update_and_attend.

    def __init__(self, n_layers, hidden, n_heads, n_kv_heads, head_dim, vocab):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        self.wq = torch.nn.ModuleList(
            torch.nn.Linear(hidden, q_dim, bias=False) for _ in range(n_layers)
        )
        self.wk = torch.nn.ModuleList(
            torch.nn.Linear(hidden, kv_dim, bias=False) for _ in range(n_layers)
        )
        self.wv = torch.nn.ModuleList(
            torch.nn.Linear(hidden, kv_dim, bias=False) for _ in range(n_layers)
        )
        self.wo = torch.nn.ModuleList(
            torch.nn.Linear(q_dim, hidden, bias=False) for _ in range(n_layers)
        )
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def _proj(self, layer_id, x):
        b, s, _ = x.shape
        q = self.wq[layer_id](x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.wk[layer_id](x)
            .view(b, s, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.wv[layer_id](x)
            .view(b, s, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        return q, k, v

    def forward(self, x, position, logits_indices):
        b, s, _ = x.shape
        for layer_id in range(self.n_layers):
            q, k, v = self._proj(layer_id, x)
            attn = update_and_attend(
                q, k, v, position, layer_id, self.scale, torch.float32
            )
            attn = attn.transpose(1, 2).reshape(b, s, -1)
            x = x + self.wo[layer_id](attn)
        h = x[:, logits_indices, :]
        return self.lm_head(h)

    def reference_forward(self, x, logits_indices):
        """Cacheless full causal attention baseline."""
        b, s, _ = x.shape
        for layer_id in range(self.n_layers):
            q, k, v = self._proj(layer_id, x)
            rep = self.n_heads // self.n_kv_heads
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            attn = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=True, scale=self.scale
            )
            attn = attn.transpose(1, 2).reshape(b, s, -1)
            x = x + self.wo[layer_id](attn)
        h = x[:, logits_indices, :]
        return self.lm_head(h)


def _positions(start, length):
    return torch.arange(start, start + length, dtype=torch.long).unsqueeze(-1)


class UpdateAndAttendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n_layers = 3
        self.hidden = 32
        self.n_heads = 4
        self.n_kv_heads = 2
        self.head_dim = 8
        self.vocab = 40
        self.model = TinyAttentionModel(
            self.n_layers,
            self.hidden,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.vocab,
        ).eval()
        self.cache_key = "test"

    def tearDown(self):
        REGISTRY.uninstall(self.cache_key)

    def _config(self, sizing, capacity):
        return CacheConfig(
            n_layers=self.n_layers,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            sizing=sizing,
            capacity=capacity,
        )

    def _export(self, seq_len):
        x = torch.randn(1, seq_len, self.hidden)
        pos = _positions(0, seq_len)
        idx = torch.arange(seq_len, dtype=torch.long)
        ep = torch.export.export(self.model, (x, pos, idx), strict=True)
        # ET always functionalizes; run it here (empty decomp table = functionalize
        # only) so tests catch functionalization failures plain export would miss.
        return ep.run_decompositions({})

    def test_graph_is_functional(self):
        # Export needs no cache: _export installs none, so the op traces via its
        # fake kernel only -- the cleanest statement that the cache is off-graph.
        ep = self._export(seq_len=5)

        # The model carries zero cache state: no buffer inputs ...
        buffer_inputs = [
            s for s in ep.graph_signature.input_specs if s.kind.name == "BUFFER"
        ]
        self.assertEqual(buffer_inputs, [])
        # ... and no buffer mutations in the outputs.
        mutated = [
            s
            for s in ep.graph_signature.output_specs
            if s.kind.name == "BUFFER_MUTATION"
        ]
        self.assertEqual(mutated, [])
        op_calls = [
            n
            for n in ep.graph_module.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.kvcache.update_and_attend.default
        ]
        self.assertEqual(len(op_calls), self.n_layers)

    def test_prefill_matches_baseline(self):
        seq_len = 6
        x = torch.randn(1, seq_len, self.hidden)
        ref = self.model.reference_forward(x, torch.arange(seq_len))

        ep = self._export(seq_len)
        for sizing, cap in [
            (CacheSizing.DYNAMIC, seq_len),
            (CacheSizing.STATIC, seq_len),
        ]:
            with self.subTest(sizing=sizing):
                cache = SequenceReferenceCache(self._config(sizing, cap))
                REGISTRY.install(self.cache_key, cache)
                with REGISTRY.active(self.cache_key):
                    out = ep.module()(x, _positions(0, seq_len), torch.arange(seq_len))
                torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_incremental_decode_matches_baseline(self):
        prefill_len = 4
        total = prefill_len + 3
        x_full = torch.randn(1, total, self.hidden)
        ref = self.model.reference_forward(x_full, torch.arange(total))

        ep_prefill = self._export(prefill_len)
        ep_decode = self._export(1)

        for sizing, cap in [
            (CacheSizing.DYNAMIC, total),
            (CacheSizing.STATIC, total),
        ]:
            with self.subTest(sizing=sizing):
                cache = SequenceReferenceCache(self._config(sizing, cap))
                REGISTRY.install(self.cache_key, cache)
                with REGISTRY.active(self.cache_key):
                    ep_prefill.module()(
                        x_full[:, :prefill_len, :],
                        _positions(0, prefill_len),
                        torch.arange(prefill_len),
                    )
                    for step in range(prefill_len, total):
                        out = ep_decode.module()(
                            x_full[:, step : step + 1, :],
                            _positions(step, 1),
                            torch.tensor([0], dtype=torch.long),
                        )
                        torch.testing.assert_close(
                            out[:, 0, :], ref[:, step, :], atol=1e-4, rtol=1e-4
                        )

    def test_chunked_prefill_matches_baseline(self):
        # Prefill in chunks: each chunk's queries must attend every earlier
        # chunk's keys, which holds only if CAUSAL is lower-right aligned.
        chunk, total = 3, 6
        x = torch.randn(1, total, self.hidden)
        ref = self.model.reference_forward(x, torch.arange(total))

        ep = self._export(chunk)
        cache = SequenceReferenceCache(self._config(CacheSizing.DYNAMIC, total))
        REGISTRY.install(self.cache_key, cache)
        with REGISTRY.active(self.cache_key):
            for start in range(0, total, chunk):
                out = ep.module()(
                    x[:, start : start + chunk, :],
                    _positions(start, chunk),
                    torch.arange(chunk),
                )
        torch.testing.assert_close(
            out, ref[:, total - chunk :, :], atol=1e-4, rtol=1e-4
        )

    def test_static_overflow_raises(self):
        ep = self._export(seq_len=5)
        cache = SequenceReferenceCache(self._config(CacheSizing.STATIC, capacity=3))
        REGISTRY.install(self.cache_key, cache)
        with self.assertRaises(RuntimeError), REGISTRY.active(self.cache_key):
            ep.module()(
                torch.randn(1, 5, self.hidden), _positions(0, 5), torch.arange(5)
            )

    def test_the_specs_must_cover_every_query_token(self):
        # Each spec's queries are placed by the running total of the ones
        # before it, so a cache that miscounts would attend the wrong slice
        # rather than fail. Only this check separates the two.
        class Miscounting:
            def __init__(self, q_len):
                self.q_len = q_len

            def update_and_fetch(self, layer_id, k, v, position):
                return [AttendSpec(k=k, v=v, kind=MaskKind.NONE, q_len=self.q_len)]

        q = torch.randn(1, self.n_heads, 3, self.head_dim)
        kv = torch.randn(1, self.n_kv_heads, 3, self.head_dim)
        for q_len in (2, 4):  # answering too few, and claiming too many
            with self.subTest(q_len=q_len):
                REGISTRY.install(self.cache_key, Miscounting(q_len))
                with self.assertRaisesRegex(ValueError, "of 3 query tokens"):
                    with REGISTRY.active(self.cache_key):
                        update_and_attend(
                            q, kv, kv, _positions(0, 3), 0, 0.125, torch.float32
                        )

    def test_output_shape_uses_value_head_dim(self):
        # The output's last dim comes from v, which may differ from q's head dim
        # (e.g. MLA). Export (fake kernel only) and check the op node's meta.
        class OneCall(torch.nn.Module):
            def forward(self, q, k, v, position):
                return update_and_attend(q, k, v, position, 0, 0.125, torch.float32)

        q = torch.randn(1, 4, 3, 8)
        k = torch.randn(1, 4, 3, 8)
        v = torch.randn(1, 4, 3, 5)  # value head dim (5) != q/k head dim (8)
        ep = torch.export.export(OneCall(), (q, k, v, _positions(0, 3)), strict=True)
        node = next(
            n
            for n in ep.graph_module.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.kvcache.update_and_attend.default
        )
        self.assertEqual(tuple(node.meta["val"].shape), (1, 4, 3, 5))


class BatchedSequenceCacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n_layers, self.hidden = 2, 16
        self.n_heads, self.n_kv_heads, self.head_dim = 4, 2, 8
        self.model = TinyAttentionModel(
            self.n_layers,
            self.hidden,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            40,
        ).eval()
        self.cache_key = "batched-sequences"

    def tearDown(self):
        REGISTRY.uninstall(self.cache_key)

    def _cache(self, capacity=16, layers=None, max_context=None):
        cache = BatchedSequenceReferenceCache(
            CacheConfig(
                n_layers=self.n_layers,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                capacity=capacity,
                layers=[LayerPolicy.flat()] if layers is None else layers,
                max_context=max_context,
            )
        )
        REGISTRY.install(self.cache_key, cache)
        return cache

    def _step(self, cache, x, positions, seq_ids):
        cache.declare_step(seq_ids)
        with REGISTRY.active(self.cache_key):
            return self.model(x, positions, torch.arange(x.shape[1]))

    def _attention_inputs(self, length):
        return (
            torch.randn(1, self.n_heads, length, self.head_dim),
            torch.randn(1, self.n_kv_heads, length, self.head_dim),
            torch.randn(1, self.n_kv_heads, length, self.head_dim),
            _positions(0, length),
        )

    def _attend(self, cache, inputs, layer_id=0):
        # What the op does: fetch one spec per span, attend each over the query
        # tokens it answers, rejoin.
        q, k, v, positions = inputs
        specs = cache.update_and_fetch(layer_id, k, v, positions)
        outputs, start = [], 0
        for spec in specs:
            end = start + spec.q_len
            outputs.append(
                attend(
                    q[:, :, start:end, :],
                    spec,
                    self.head_dim**-0.5,
                    torch.float32,
                )
            )
            start = end
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)

    def test_single_span_matches_single_sequence(self):
        x = torch.randn(1, 5, self.hidden)
        out = self._step(self._cache(), x, _positions(0, 5), [3] * 5)
        torch.testing.assert_close(
            out,
            self.model.reference_forward(x, torch.arange(5)),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_multiple_sequence_spans_match_separate_runs(self):
        a = torch.randn(1, 4, self.hidden)
        b = torch.randn(1, 3, self.hidden)
        tokens, positions, seq_ids, _ = flatten_step({2: (a, 0), 7: (b, 0)})

        out = self._step(self._cache(), tokens, positions, seq_ids)

        torch.testing.assert_close(
            out[:, :4],
            self.model.reference_forward(a, torch.arange(4)),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            out[:, 4:],
            self.model.reference_forward(b, torch.arange(3)),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_repeated_sequence_spans_preserve_input_order(self):
        a = torch.randn(1, 3, self.hidden)
        b = torch.randn(1, 1, self.hidden)
        tokens = torch.cat([a[:, :2], b, a[:, 2:]], dim=1)
        positions = torch.tensor([[0], [1], [0], [2]], dtype=torch.long)
        out = self._step(self._cache(), tokens, positions, [2, 2, 7, 2])

        a_out = self.model.reference_forward(a, torch.arange(3))
        b_out = self.model.reference_forward(b, torch.arange(1))
        torch.testing.assert_close(out[:, [0, 1, 3]], a_out, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(out[:, 2:3], b_out, atol=1e-4, rtol=1e-4)

    def test_decode_continues_each_private_sequence(self):
        a = torch.randn(1, 4, self.hidden)
        b = torch.randn(1, 3, self.hidden)
        cache = self._cache()

        tokens, positions, seq_ids, _ = flatten_step(
            {2: (a[:, :3], 0), 7: (b[:, :2], 0)}
        )
        self._step(cache, tokens, positions, seq_ids)

        tokens, positions, seq_ids, _ = flatten_step(
            {2: (a[:, 3:], 3), 7: (b[:, 2:], 2)}
        )
        out = self._step(cache, tokens, positions, seq_ids)

        torch.testing.assert_close(
            out[:, 0],
            self.model.reference_forward(a, torch.arange(4))[:, -1],
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            out[:, 1],
            self.model.reference_forward(b, torch.arange(3))[:, -1],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_a_span_must_continue_its_own_sequence(self):
        # A private history appends at its length, so a wrong position would
        # still land contiguously. Only this check separates the two.
        cache = self._cache()
        q, k, v, _ = self._attention_inputs(2)
        cache.declare_step([5, 5])

        with self.assertRaisesRegex(ValueError, r"declares \[1, 2\], not \[0, 1\]"):
            self._attend(cache, (q, k, v, _positions(1, 2)))
        self.assertEqual(cache.seq_len(5), 0)  # a refusal writes nothing

        # Ascending is not enough; a span is a consecutive run.
        gapped = torch.tensor([[0], [2]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, r"declares \[0, 2\], not \[0, 1\]"):
            self._attend(cache, (q, k, v, gapped))
        self.assertEqual(cache.seq_len(5), 0)

        # The declaration still stands, so the same layer can be retried.
        self._attend(cache, (q, k, v, _positions(0, 2)))
        self.assertEqual(cache.seq_len(5), 2)

    def test_a_sequence_spanned_twice_continues_across_both(self):
        cache = self._cache()
        q, k, v, _ = self._attention_inputs(3)

        # Tokens 0 and 2 are seq 4, token 1 is seq 9; seq 4's second span picks
        # up where its first left off rather than at its prior length.
        cache.declare_step([4, 9, 4])
        self._attend(cache, (q, k, v, torch.tensor([[0], [0], [1]])))
        self.assertEqual(cache.seq_len(4), 2)
        self.assertEqual(cache.seq_len(9), 1)

        # The bad position is in the last span, so a per-span check would have
        # written the first two before refusing.
        cache.declare_step([4, 9, 4])
        with self.assertRaisesRegex(ValueError, r"holds 3 .*declares \[4\], not \[3\]"):
            self._attend(cache, (q, k, v, torch.tensor([[2], [1], [4]])))
        self.assertEqual(cache.seq_len(4), 2)
        self.assertEqual(cache.seq_len(9), 1)

    def test_requires_one_declared_step_per_forward(self):
        cache = self._cache()
        inputs = self._attention_inputs(1)

        with self.assertRaisesRegex(RuntimeError, "no step declared"):
            self._attend(cache, inputs)

        cache.declare_step([2])
        self._attend(cache, inputs)
        with self.assertRaisesRegex(RuntimeError, "served twice"):
            self._attend(cache, inputs)

    def test_declaration_and_sequence_verbs_validate_ids(self):
        cache = self._cache()
        with self.assertRaisesRegex(ValueError, "at least one token"):
            cache.declare_step([])

        for call in (
            lambda: cache.declare_step([-1]),
            lambda: cache.seq_rm(-1),
            lambda: cache.seq_len(-1),
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

        # Private histories are dict entries, so nothing caps the id.
        cache.declare_step([9999])
        self.assertEqual(cache.seq_len(9999), 0)

    def test_capacity_is_the_pool_total_and_refusal_changes_nothing(self):
        cache = self._cache(capacity=4)
        with self.assertRaisesRegex(RuntimeError, "exceeds capacity"):
            cache.declare_step([2] * 5)
        self.assertEqual(cache.seq_len(2), 0)

        # Two sequences share the budget rather than each getting one.
        a = torch.randn(1, 2, self.hidden)
        b = torch.randn(1, 2, self.hidden)
        tokens, positions, seq_ids, _ = flatten_step({2: (a, 0), 7: (b, 0)})
        self._step(cache, tokens, positions, seq_ids)
        self.assertEqual(cache.seq_len(2), 2)
        self.assertEqual(cache.seq_len(7), 2)

        # Either sequence is now blocked by what the other holds.
        with self.assertRaisesRegex(RuntimeError, "exceeds capacity"):
            cache.declare_step([2])
        self.assertEqual(cache.seq_len(2), 2)
        self.assertEqual(cache.seq_len(7), 2)

    def test_forward_width_must_match_tensors_and_declaration(self):
        one = self._attention_inputs(1)
        two = self._attention_inputs(2)
        cases = (
            ("position", [2], (one[0], one[1], one[2], two[3]), "same token count"),
            ("q/k", [2, 2], (two[0], one[1], one[2], two[3]), "same token count"),
            ("k/v", [2], (one[0], one[1], two[2], one[3]), "same token count"),
            ("declaration", [2, 2], one, "must match declare_step"),
        )
        for name, seq_ids, inputs, message in cases:
            with self.subTest(name=name):
                cache = self._cache()
                cache.declare_step(seq_ids)
                with self.assertRaisesRegex(ValueError, message):
                    self._attend(cache, inputs)

    def test_each_sequence_is_bounded_by_the_model_context(self):
        cache = self._cache(capacity=16, max_context=3)
        self.assertTrue(cache.can_admit(0, 3))
        self.assertFalse(cache.can_admit(0, 4))  # past the trained positions
        # It speaks only for the sequence: 16 cells free, still refused at 4.
        self.assertFalse(cache.can_admit(0, 16))

        pos = torch.tensor([[0], [1], [0]], dtype=torch.long)
        self._step(cache, torch.randn(1, 3, self.hidden), pos, [0, 0, 1])
        self.assertTrue(cache.can_admit(0, 1))  # 0 has reached 2 of 3
        self.assertFalse(cache.can_admit(0, 2))
        self.assertTrue(cache.can_admit(1, 2))  # bounded independently

        with self.assertRaisesRegex(RuntimeError, "model context limit"):
            cache.declare_step([0, 0])
        cache.declare_step([0, 1])

    def test_admission_reports_reach_not_room(self):
        # No max_context: nothing bounds a sequence's reach.
        cache = self._cache(capacity=4)
        self.assertTrue(cache.can_admit(0, 1000))

        pos = torch.tensor([[0], [0]], dtype=torch.long)
        self._step(cache, torch.randn(1, 2, self.hidden), pos, [0, 1])
        # Each could take two more alone; together they overrun four cells.
        self.assertTrue(cache.can_admit(0, 2))
        self.assertTrue(cache.can_admit(1, 2))
        with self.assertRaises(RuntimeError):
            cache.declare_step([0, 0, 1, 1])
        cache.declare_step([0, 1])

    def test_admission_ignores_a_full_pool(self):
        cache = self._cache(capacity=2, max_context=100)
        self._step(cache, torch.randn(1, 2, self.hidden), _positions(0, 2), [0] * 2)
        # Room is gone, reach is not.
        self.assertTrue(cache.can_admit(0, 1))
        with self.assertRaises(RuntimeError):
            cache.declare_step([0])

    def test_sequence_removal_invalidates_a_declared_step(self):
        cache = self._cache()
        cache.declare_step([2])
        cache.seq_rm(2)

        self.assertEqual(cache.seq_len(2), 0)
        with self.assertRaisesRegex(RuntimeError, "no step declared"):
            self._attend(cache, self._attention_inputs(1))

    def test_rewind_truncates_and_seq_rm_drops_the_sequence(self):
        cache = self._cache()
        self._step(cache, torch.randn(1, 4, self.hidden), _positions(0, 4), [1] * 4)
        self._step(cache, torch.randn(1, 2, self.hidden), _positions(0, 2), [6] * 2)

        cache.rewind(1, 2)  # keep positions 0..1
        self.assertEqual(cache.seq_len(1), 2)
        self.assertEqual(cache.seq_len(6), 2)  # its neighbour is untouched

        cache.seq_rm(1)  # the whole sequence
        self.assertEqual(cache.seq_len(1), 0)
        self.assertEqual(cache.seq_len(6), 2)

    def test_rewinding_then_continuing_matches_an_unbroken_run(self):
        x = torch.randn(1, 5, self.hidden)
        ref = self.model.reference_forward(x, torch.arange(5))

        cache = self._cache()
        self._step(cache, x[:, :4], _positions(0, 4), [3] * 4)
        cache.rewind(3, 2)  # discard positions 2..3
        out = self._step(cache, x[:, 2:], _positions(2, 3), [3] * 3)

        torch.testing.assert_close(out, ref[:, 2:], atol=1e-4, rtol=1e-4)

    def test_rewind_refuses_to_grow_or_pass_a_window(self):
        cache = self._cache(layers=[LayerPolicy.ring(2)])
        self._step(cache, torch.randn(1, 5, self.hidden), _positions(0, 5), [0] * 5)

        with self.assertRaisesRegex(ValueError, "the history holds 5"):
            cache.rewind(0, 6)
        # A windowed layer keeps only its last two positions, so 3 is the floor
        # even though this reference still holds the older ones.
        with self.assertRaisesRegex(ValueError, "retains only from 3"):
            cache.rewind(0, 1)
        cache.rewind(0, 3)
        self.assertEqual(cache.seq_len(0), 3)


class CellCacheTest(unittest.TestCase):
    # Many sequences over one pool of per-token cells, flat on the token axis.
    # The baseline throughout is the cacheless model: whatever a sequence would
    # have computed alone, it must still compute when batched beside others.

    CAPACITY = 32

    def setUp(self):
        torch.manual_seed(0)
        self.n_layers, self.hidden = 2, 16
        self.n_heads, self.n_kv_heads, self.head_dim = 4, 2, 8
        self.model = TinyAttentionModel(
            self.n_layers,
            self.hidden,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            40,
        ).eval()
        self.cache_key = "cells"

    def tearDown(self):
        REGISTRY.uninstall(self.cache_key)

    def _cache(
        self,
        capacity=CAPACITY,
        sizing=CacheSizing.DYNAMIC,
        layers=None,
        n_layers=None,
        max_context=None,
    ):
        cache = CellReferenceCache(
            CacheConfig(
                n_layers=self.n_layers if n_layers is None else n_layers,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                capacity=capacity,
                sizing=sizing,
                layers=[LayerPolicy.flat()] if layers is None else layers,
                max_context=max_context,
            )
        )
        REGISTRY.install(self.cache_key, cache)
        return cache

    def _step(self, cache, x, positions, seqs):
        """One forward carrying `x`, whose tokens have these positions/seqs."""
        cache.declare_step(seqs)
        pos = torch.tensor(positions, dtype=torch.long).unsqueeze(-1)
        with REGISTRY.active(self.cache_key):
            return self.model(x, pos, torch.arange(x.shape[1]))

    def test_single_sequence_matches_baseline(self):
        x = torch.randn(1, 5, self.hidden)
        out = self._step(self._cache(), x, list(range(5)), [0] * 5)
        torch.testing.assert_close(
            out, self.model.reference_forward(x, torch.arange(5)), atol=1e-4, rtol=1e-4
        )

    def test_batched_sequences_match_separate_runs(self):
        # Two prefills in ONE forward. Each must equal what it computes alone,
        # which is exactly the isolation the per-cell seq bitset buys.
        a, b = torch.randn(1, 4, self.hidden), torch.randn(1, 3, self.hidden)
        cache = self._cache()

        # {seq_id: (tokens, start_pos)} -> the step's parallel arrays
        tokens, positions, seq_ids, _ = flatten_step({0: (a, 0), 1: (b, 0)})
        cache.declare_step(seq_ids)
        with REGISTRY.active(self.cache_key):
            # every row, not one per sequence: each token is compared below
            out = self.model(tokens, positions, torch.arange(tokens.shape[1]))

        torch.testing.assert_close(
            out[:, :4, :],
            self.model.reference_forward(a, torch.arange(4)),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            out[:, 4:, :],
            self.model.reference_forward(b, torch.arange(3)),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_batched_decode_continues_each_sequence(self):
        # Prefill both, then one forward carrying a new token for each, laid
        # out by flatten_step -- one logits row per sequence, not per token.
        a, b = torch.randn(1, 4, self.hidden), torch.randn(1, 3, self.hidden)
        cache = self._cache()

        tokens, positions, seq_ids, logits_indices = flatten_step(
            {0: (a[:, :3], 0), 1: (b[:, :2], 0)}
        )
        cache.declare_step(seq_ids)
        with REGISTRY.active(self.cache_key):
            self.model(tokens, positions, logits_indices)

        tokens, positions, seq_ids, logits_indices = flatten_step(
            {0: (a[:, 3:], 3), 1: (b[:, 2:], 2)}
        )
        cache.declare_step(seq_ids)
        with REGISTRY.active(self.cache_key):
            out = self.model(tokens, positions, logits_indices)

        torch.testing.assert_close(
            out[:, 0, :],
            self.model.reference_forward(a, torch.arange(4))[:, -1, :],
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            out[:, 1, :],
            self.model.reference_forward(b, torch.arange(3))[:, -1, :],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_fork_shares_cells_and_history(self):
        trunk, tail = torch.randn(1, 4, self.hidden), torch.randn(1, 1, self.hidden)
        cache = self._cache()
        self._step(cache, trunk, [0, 1, 2, 3], [0] * 4)

        free_before = cache.free_cells()
        cache.seq_cp(0, 1)
        self.assertEqual(cache.free_cells(), free_before)  # no cell, no byte copied
        self.assertEqual(cache.seq_len(1), 4)

        out = self._step(cache, tail, [4], [1])  # the branch continues the trunk
        torch.testing.assert_close(
            out[:, 0, :],
            self.model.reference_forward(
                torch.cat([trunk, tail], dim=1), torch.arange(5)
            )[:, -1, :],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_seq_rm_frees_only_unowned_cells(self):
        cache = self._cache()
        self._step(cache, torch.randn(1, 3, self.hidden), [0, 1, 2], [0] * 3)
        cache.seq_cp(0, 1)

        cache.seq_rm(0)
        self.assertEqual(cache.seq_len(0), 0)
        self.assertEqual(cache.seq_len(1), 3)  # the fork still owns them
        self.assertEqual(cache.free_cells(), self.CAPACITY - 3)

        cache.seq_rm(1)
        self.assertEqual(cache.free_cells(), self.CAPACITY)

    def test_admission_reports_reach_not_room(self):
        cache = self._cache(capacity=4)
        self.assertTrue(cache.can_admit(0, 1000))

        self._step(cache, torch.randn(1, 2, self.hidden), [0, 0], [0, 1])
        self.assertTrue(cache.can_admit(0, 2))
        self.assertTrue(cache.can_admit(1, 2))
        with self.assertRaises(RuntimeError):
            cache.declare_step([0, 0, 1, 1])
        cache.declare_step([0, 1])

    def test_admission_ignores_a_full_pool(self):
        cache = self._cache(capacity=2, max_context=100)
        self._step(cache, torch.randn(1, 2, self.hidden), [0, 1], [0, 0])
        self.assertTrue(cache.can_admit(0, 1))
        with self.assertRaises(RuntimeError):
            cache.declare_step([0])

    def test_flatten_step_lays_out_the_parallel_arrays(self):
        tokens, positions, seq_ids, logits_indices = flatten_step(
            {
                0: (torch.zeros(1, 3, self.hidden), 5),
                1: (torch.ones(1, 2, self.hidden), 0),
            }
        )
        self.assertEqual(tokens.shape[1], 5)  # one axis, both sequences
        self.assertEqual(positions.squeeze(-1).tolist(), [5, 6, 7, 0, 1])
        self.assertEqual(seq_ids, [0, 0, 0, 1, 1])
        self.assertEqual(logits_indices.tolist(), [2, 4])  # each sequence's last

    def test_fork_at_a_position_shares_only_the_prefix(self):
        cache = self._cache()
        self._step(cache, torch.randn(1, 4, self.hidden), [0, 1, 2, 3], [0] * 4)

        cache.seq_cp(0, 1, upto=2)
        self.assertEqual(cache.seq_len(0), 4)
        self.assertEqual(cache.seq_len(1), 2)  # only positions 0 and 1
        self.assertEqual(cache.free_cells(), self.CAPACITY - 4)  # still no copy

    def test_freeing_the_tail_shrinks_the_read_window(self):
        cache = self._cache()
        kv = torch.randn(1, self.n_kv_heads, 4, self.head_dim)
        cache.declare_step([0] * 4)
        spec = cache.update_and_fetch(0, kv, kv, _positions(0, 4))[0]
        self.assertEqual(spec.k.shape[2], 4)  # four cells held, so a window of four

        cache.seq_rm(0)  # frees all four, so used_end walks back to 0
        self.assertEqual(cache.free_cells(), self.CAPACITY)

        # one token reclaims cell 0, so the window is its own single cell
        kv = torch.randn(1, self.n_kv_heads, 1, self.head_dim)
        cache.declare_step([1])
        spec = cache.update_and_fetch(0, kv, kv, torch.tensor([[0]]))[0]
        self.assertEqual(spec.k.shape[2], 1)  # the window length is 1, not the old 4
        self.assertEqual(spec.mask.shape[-1], 1)

    def test_rewind_frees_only_the_tail(self):
        cache = self._cache()
        self._step(cache, torch.randn(1, 5, self.hidden), [0, 1, 2, 3, 4], [0] * 5)

        cache.rewind(0, 4)  # backtrack: drop position 4 onwards
        self.assertEqual(cache.seq_len(0), 4)
        self.assertEqual(cache.free_cells(), self.CAPACITY - 4)

        cache.rewind(0, 2)
        self.assertEqual(cache.seq_len(0), 2)
        self.assertEqual(cache.free_cells(), self.CAPACITY - 2)

    def test_every_verb_range_checks_the_seq_id(self):
        # An id past the bitset would set a bit no int64 can hold, surfacing
        # much later as an overflow while building the mask.
        cache = self._cache()
        for call in (
            lambda: cache.declare_step([MAX_SEQS]),
            lambda: cache.seq_cp(0, MAX_SEQS),
            lambda: cache.seq_cp(MAX_SEQS, 0),
            lambda: cache.seq_rm(MAX_SEQS),
            lambda: cache.seq_len(MAX_SEQS),
            lambda: cache.seq_len(-1),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_layer_policy_rejects_a_mismatched_window(self):
        # The kind carries the meaning, so a window on a flat layer -- or a ring
        # layer without one -- is a config error rather than a silent no-op.
        with self.assertRaises(ValueError):
            LayerPolicy(kind=LayerKind.FLAT, window=4)
        with self.assertRaises(ValueError):
            LayerPolicy(kind=LayerKind.RING, window=0)
        with self.assertRaises(ValueError):
            LayerPolicy.ring(-1)
        with self.assertRaises(ValueError):  # one policy, or one per layer
            self._cache(
                layers=[LayerPolicy.flat(), LayerPolicy.ring(2), LayerPolicy.ring(2)]
            )

    def test_window_narrows_each_query_without_crossing_sequences(self):
        cache = self._cache(layers=[LayerPolicy.ring(2)])
        kv = torch.randn(1, self.n_kv_heads, 4, self.head_dim)
        cache.declare_step([0] * 4)
        spec = cache.update_and_fetch(0, kv, kv, _positions(0, 4))[0]

        # offsets[i][j] = j - i, so <= 0 is causal and > -2 keeps the newest
        # two: a band whose row 2 drops key 0, which plain causal would keep.
        offsets = torch.arange(4) - torch.arange(4).unsqueeze(-1)
        torch.testing.assert_close(spec.mask, (offsets <= 0) & (offsets > -2))

        # a second sequence is bounded the same way, and still sees none of
        # the first's cells even though they are inside its window
        cache.declare_step([1, 1])
        kv = torch.randn(1, self.n_kv_heads, 2, self.head_dim)
        spec = cache.update_and_fetch(0, kv, kv, _positions(0, 2))[0]
        expected = torch.zeros(2, 6, dtype=torch.bool)
        expected[0, 4] = expected[1, 4] = expected[1, 5] = True
        torch.testing.assert_close(spec.mask, expected)

    def test_layers_can_window_independently(self):
        # One cell table, but layers may disagree about the window: the mask is
        # per policy, not per layer, so a mixed model costs one extra mask.
        cache = self._cache(layers=[LayerPolicy.flat(), LayerPolicy.ring(2)])
        kv = torch.randn(1, self.n_kv_heads, 4, self.head_dim)
        cache.declare_step([0] * 4)
        pos = _positions(0, 4)
        flat = cache.update_and_fetch(0, kv, kv, pos)[0].mask
        windowed = cache.update_and_fetch(1, kv, kv, pos)[0].mask

        offsets = torch.arange(4) - torch.arange(4).unsqueeze(-1)
        torch.testing.assert_close(flat, offsets <= 0)
        torch.testing.assert_close(windowed, (offsets <= 0) & (offsets > -2))

    def test_layers_sharing_a_window_share_one_mask(self):
        # The mask is per policy: two windowed layers get the same object, and
        # the flat one a different mask, so a mixed model costs one extra.
        cache = self._cache(
            layers=[LayerPolicy.flat(), LayerPolicy.ring(2), LayerPolicy.ring(2)],
            n_layers=3,
        )
        kv = torch.randn(1, self.n_kv_heads, 4, self.head_dim)
        cache.declare_step([0] * 4)
        pos = _positions(0, 4)
        flat = cache.update_and_fetch(0, kv, kv, pos)[0].mask
        first = cache.update_and_fetch(1, kv, kv, pos)[0].mask
        second = cache.update_and_fetch(2, kv, kv, pos)[0].mask

        self.assertIs(first, second)
        self.assertIsNot(flat, first)

    def test_windowed_decode_attends_only_the_retained_cells(self):
        window = 2
        cache = self._cache(layers=[LayerPolicy.ring(window)])
        kv = torch.randn(1, self.n_kv_heads, 4, self.head_dim)
        cache.declare_step([0] * 4)
        cache.update_and_fetch(0, kv, kv, _positions(0, 4))

        cache.declare_step([0])
        kv = torch.randn(1, self.n_kv_heads, 1, self.head_dim)
        spec = cache.update_and_fetch(0, kv, kv, _positions(4, 1))[0]

        q = torch.randn(1, self.n_heads, 1, self.head_dim)
        scale = self.head_dim**-0.5
        torch.testing.assert_close(
            attend(q, spec, scale, torch.float32),
            attend(  # the last `window` cells of its sequence, unmasked
                q,
                AttendSpec(
                    k=spec.k[:, :, -window:, :],
                    v=spec.v[:, :, -window:, :],
                    kind=MaskKind.NONE,
                    q_len=1,
                ),
                scale,
                torch.float32,
            ),
        )

    def test_admission_fails_before_the_forward(self):
        cache = self._cache(capacity=4)
        self.assertEqual(cache.free_cells(), 4)
        with self.assertRaises(RuntimeError):
            cache.declare_step([0] * 5)

    def test_each_sequence_is_bounded_by_the_model_context(self):
        cache = self._cache(capacity=16, max_context=3)
        self.assertTrue(cache.can_admit(0, 3))
        self.assertFalse(cache.can_admit(0, 4))  # past the trained positions
        # It speaks only for the sequence: 16 cells free, still refused at 4.
        self.assertFalse(cache.can_admit(0, 16))

        self._step(cache, torch.randn(1, 3, self.hidden), [0, 1, 0], [0, 0, 1])
        self.assertTrue(cache.can_admit(0, 1))  # 0 has reached 2 of 3
        self.assertFalse(cache.can_admit(0, 2))
        self.assertTrue(cache.can_admit(1, 2))  # bounded independently

        with self.assertRaisesRegex(RuntimeError, "model context limit"):
            cache.declare_step([0, 0])
        cache.declare_step([0, 1])

    def test_step_protocol_is_enforced(self):
        cache = self._cache()
        kv = torch.randn(1, self.n_kv_heads, 1, self.head_dim)
        pos = torch.tensor([[0]])

        with self.assertRaises(ValueError):  # a step with no tokens
            cache.declare_step([])

        cache.declare_step([0, 0])  # declares two tokens, forward carries one
        with self.assertRaises(ValueError):
            cache.update_and_fetch(0, kv, kv, pos)
        with self.assertRaises(RuntimeError):  # the failed attempt still cleared it
            cache.update_and_fetch(0, kv, kv, torch.tensor([[0], [1]]))

        cache.declare_step([0])
        cache.update_and_fetch(0, kv, kv, pos)
        with self.assertRaises(RuntimeError):  # a second step, no declare_step
            cache.update_and_fetch(0, kv, kv, pos)

    def test_growth_keeps_cell_indices_and_bytes(self):
        # A grown pool must append rows only: a cell's index is its name, held
        # by the plan and by _pos/_seq, so renumbering or dropping rows would
        # move history without anything noticing.
        cache = self._cache()
        first = torch.randn(1, self.n_kv_heads, 2, self.head_dim)
        cache.declare_step([0, 0])
        spec = cache.update_and_fetch(0, first, first, torch.tensor([[0], [1]]))[0]
        self.assertEqual(spec.k.shape[2], 2)  # a short session reserves a short pool

        rest = torch.randn(1, self.n_kv_heads, 6, self.head_dim)
        cache.declare_step([0] * 6)
        spec = cache.update_and_fetch(
            0, rest, rest, torch.tensor([[p] for p in range(2, 8)])
        )[0]
        self.assertEqual(spec.k.shape[2], 8)
        # cells 0,1 unmoved
        torch.testing.assert_close(spec.k[:, :, :2, :], first)
        torch.testing.assert_close(spec.v[:, :, 2:, :], rest)

    def test_sizings_agree(self):
        x = torch.randn(1, 5, self.hidden)
        out = [
            self._step(self._cache(sizing=s), x, list(range(5)), [0] * 5)
            for s in (CacheSizing.DYNAMIC, CacheSizing.STATIC)
        ]
        torch.testing.assert_close(out[0], out[1])

    def test_a_verb_does_not_hide_a_missing_declare_step(self):
        # A sequence verb drops the memoized plan, which must not be mistaken
        # for the start of a step -- that would silently reuse the previous
        # step's sequence assignment for the new tokens.
        cache = self._cache()
        kv = torch.randn(1, self.n_kv_heads, 2, self.head_dim)
        pos = torch.tensor([[0], [0]])
        cache.declare_step([0, 1])
        cache.update_and_fetch(0, kv, kv, pos)

        cache.seq_rm(2)  # any verb; a no-op here beyond dropping the plan
        with self.assertRaises(RuntimeError):  # layer 0 was already served
            cache.update_and_fetch(0, kv, kv, pos)
        with self.assertRaises(RuntimeError):  # and the declaration is spent
            cache.update_and_fetch(1, kv, kv, pos)


class SequenceSpecTest(unittest.TestCase):
    # Which mask semantic the cache declares for each shape of step.

    def setUp(self):
        torch.manual_seed(0)
        self.cache = SequenceReferenceCache(
            CacheConfig(n_layers=1, n_kv_heads=2, head_dim=4, capacity=8)
        )

    def _update(self, start, q_len):
        kv = torch.randn(1, 2, q_len, 4)
        return self.cache.update_and_fetch(0, kv, kv, _positions(start, q_len))[0]

    def test_decode_is_unmasked(self):
        self.assertEqual(self._update(0, 1).kind, MaskKind.NONE)

    def test_fresh_prefill_is_causal(self):
        self.assertEqual(self._update(0, 4).kind, MaskKind.CAUSAL)

    def test_continuation_is_an_explicit_lower_right_band(self):
        self._update(0, 4)
        spec = self._update(4, 3)  # 3 new cells at the tail of a 7-cell window
        self.assertEqual(spec.kind, MaskKind.EXPLICIT)
        self.assertEqual(spec.mask.dtype, torch.bool)
        torch.testing.assert_close(
            spec.mask, torch.ones(3, 7, dtype=torch.bool).tril(7 - 3)
        )


class WindowedSpecTest(unittest.TestCase):
    # A sliding window bounds each query from below. The reference keeps the
    # whole history and masks, so what is pinned here is the semantic a ring
    # buffer has to reproduce, not the ring's addressing.

    HEADS, DIM = 2, 4

    def setUp(self):
        torch.manual_seed(0)
        self.scale = self.DIM**-0.5

    def _cache(self, policy, sizing=CacheSizing.DYNAMIC):
        return SequenceReferenceCache(
            CacheConfig(
                n_layers=1,
                n_kv_heads=self.HEADS,
                head_dim=self.DIM,
                capacity=32,
                sizing=sizing,
                layers=[policy],
            )
        )

    def _update(self, cache, n, start):
        kv = torch.randn(1, self.HEADS, n, self.DIM)
        return cache.update_and_fetch(0, kv, kv, _positions(start, n))

    def _attend(self, q, spec):
        return attend(q, spec, self.scale, torch.float32)

    def _unmasked(self, q, k, v):
        # The same queries over a hand-picked window, for the spec to match.
        spec = AttendSpec(k=k, v=v, kind=MaskKind.NONE, q_len=q.shape[-2])
        return attend(q, spec, self.scale, torch.float32)

    def test_decode_attends_only_the_window(self):
        window = 3
        cache = self._cache(LayerPolicy.ring(window))
        self._update(cache, 5, 0)
        spec = self._update(cache, 1, 5)[0]

        q = torch.randn(1, self.HEADS, 1, self.DIM)
        torch.testing.assert_close(
            self._attend(q, spec),
            self._unmasked(  # the last `window` cells, unmasked
                q, spec.k[:, :, -window:, :], spec.v[:, :, -window:, :]
            ),
        )

    def test_each_prefill_query_attends_its_own_window(self):
        window = 2
        cache = self._cache(LayerPolicy.ring(window))
        spec = self._update(cache, 4, 0)[0]
        self.assertEqual(spec.kind, MaskKind.EXPLICIT)

        q = torch.randn(1, self.HEADS, 4, self.DIM)
        out = self._attend(q, spec)
        for i in range(4):  # query at position i sees (i - window, i]
            lo = max(0, i - window + 1)
            torch.testing.assert_close(
                out[:, :, i : i + 1, :],
                self._unmasked(
                    q[:, :, i : i + 1, :],
                    spec.k[:, :, lo : i + 1, :],
                    spec.v[:, :, lo : i + 1, :],
                ),
            )

    def test_layers_can_window_independently(self):
        # gemma-style: only some layers are windowed, so one step yields two
        # different semantics from the same cache.
        cache = SequenceReferenceCache(
            CacheConfig(
                n_layers=2,
                n_kv_heads=self.HEADS,
                head_dim=self.DIM,
                capacity=32,
                layers=[LayerPolicy.flat(), LayerPolicy.ring(2)],
            )
        )
        kv = torch.randn(1, self.HEADS, 4, self.DIM)
        pos = _positions(0, 4)
        flat = cache.update_and_fetch(0, kv, kv, pos)[0]
        windowed = cache.update_and_fetch(1, kv, kv, pos)[0]

        self.assertEqual(flat.kind, MaskKind.CAUSAL)
        self.assertEqual(windowed.kind, MaskKind.EXPLICIT)
        offsets = torch.arange(4) - torch.arange(4).unsqueeze(-1)  # j - i
        torch.testing.assert_close(windowed.mask, (offsets <= 0) & (offsets > -2))

    def test_windowed_continuation_bounds_the_band_at_both_ends(self):
        # Continuation with a window: the upper bound (new cells at the tail)
        # and the lower bound (the window) are both live in the same mask.
        window = 2
        cache = self._cache(LayerPolicy.ring(window))
        self._update(cache, 4, 0)
        spec = self._update(cache, 3, 4)[0]

        self.assertEqual(spec.kind, MaskKind.EXPLICIT)
        q_len, total = 3, 7
        offsets = torch.arange(total) - torch.arange(q_len).unsqueeze(-1)
        torch.testing.assert_close(
            spec.mask,
            (offsets <= total - q_len) & (offsets > total - q_len - window),
        )

    def test_window_equal_to_history_stays_fused(self):
        # windowed is 0 < window < total, so equality is the boundary: at
        # exactly `window` cells nothing is bounded from below, and one cell
        # later the band appears.
        window = 4
        cache = self._cache(LayerPolicy.ring(window))
        self.assertEqual(self._update(cache, window, 0)[0].kind, MaskKind.CAUSAL)
        self.assertEqual(self._update(cache, 1, window)[0].kind, MaskKind.EXPLICIT)

    def test_static_sizing_windows_like_dynamic(self):
        # STATIC writes into a preallocated buffer and slices it; the window is
        # computed off the used length either way.
        window = 2
        torch.manual_seed(0)
        static = self._cache(LayerPolicy.ring(window), sizing=CacheSizing.STATIC)
        s_spec = self._update(static, 4, 0)[0]
        torch.manual_seed(0)
        dynamic = self._cache(LayerPolicy.ring(window))
        d_spec = self._update(dynamic, 4, 0)[0]

        torch.testing.assert_close(s_spec.k, d_spec.k)
        torch.testing.assert_close(s_spec.v, d_spec.v)
        self.assertEqual(s_spec.kind, d_spec.kind)
        torch.testing.assert_close(s_spec.mask, d_spec.mask)

    def test_window_wider_than_history_stays_fused(self):
        # Nothing to bound from below, so the window must not force a mask.
        for policy in (LayerPolicy.flat(), LayerPolicy.ring(64)):
            cache = self._cache(policy)
            self.assertEqual(self._update(cache, 4, 0)[0].kind, MaskKind.CAUSAL)
            self.assertEqual(self._update(cache, 1, 4)[0].kind, MaskKind.NONE)


class AttendExplicitTest(unittest.TestCase):
    # No cache emits MaskKind.EXPLICIT yet; these pin the spec's contract (bool,
    # true = attend, broadcast over batch/heads) that cell/tree caches rely on.

    def setUp(self):
        torch.manual_seed(0)
        self.q_len, self.total, self.head_dim = 3, 5, 8
        self.q = torch.randn(1, 4, self.q_len, self.head_dim)  # GQA: 4 q heads
        self.k = torch.randn(1, 2, self.total, self.head_dim)  # over 2 kv heads
        self.v = torch.randn(1, 2, self.total, self.head_dim)
        self.scale = self.head_dim**-0.5

    def _attend(self, kind, mask=None, k=None, v=None):
        spec = AttendSpec(
            k=self.k if k is None else k,
            v=self.v if v is None else v,
            kind=kind,
            q_len=self.q_len,
            mask=mask,
        )
        return attend(self.q, spec, self.scale, torch.float32)

    def test_causal_rejects_a_non_square_window(self):
        # torch's is_causal is upper-left, so it cannot serve a continuation;
        # a cache must declare EXPLICIT there rather than CAUSAL.
        with self.assertRaises(ValueError):
            self._attend(MaskKind.CAUSAL)

    def test_explicit_attends_the_true_cells(self):
        # Polarity: masking to cells {0, 2} must equal attending over just those
        # keys. The inverted convention would instead select {1, 3, 4}.
        keep = torch.tensor([0, 2])
        mask = torch.zeros(self.q_len, self.total, dtype=torch.bool)
        mask[:, keep] = True
        torch.testing.assert_close(
            self._attend(MaskKind.EXPLICIT, mask=mask),
            self._attend(
                MaskKind.NONE,
                k=self.k.index_select(2, keep),
                v=self.v.index_select(2, keep),
            ),
        )


if __name__ == "__main__":
    unittest.main()
