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
    CacheConfig,
    CacheSizing,
    CellReferenceCache,
    ContiguousReferenceCache,
    flatten_step,
    MaskKind,
    MAX_SEQS,
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
                cache = ContiguousReferenceCache(self._config(sizing, cap))
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
                cache = ContiguousReferenceCache(self._config(sizing, cap))
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
        cache = ContiguousReferenceCache(self._config(CacheSizing.DYNAMIC, total))
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
        cache = ContiguousReferenceCache(self._config(CacheSizing.STATIC, capacity=3))
        REGISTRY.install(self.cache_key, cache)
        with self.assertRaises(RuntimeError), REGISTRY.active(self.cache_key):
            ep.module()(
                torch.randn(1, 5, self.hidden), _positions(0, 5), torch.arange(5)
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

    def _cache(self, capacity=CAPACITY, sizing=CacheSizing.DYNAMIC):
        cache = CellReferenceCache(
            CacheConfig(
                n_layers=self.n_layers,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                capacity=capacity,
                sizing=sizing,
            )
        )
        REGISTRY.install(self.cache_key, cache)
        return cache

    def _step(self, cache, x, positions, seqs):
        """One forward carrying `x`, whose tokens have these positions/seqs."""
        cache.begin_step(seqs)
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
        cache.begin_step(seq_ids)
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
        cache.begin_step(seq_ids)
        with REGISTRY.active(self.cache_key):
            self.model(tokens, positions, logits_indices)

        tokens, positions, seq_ids, logits_indices = flatten_step(
            {0: (a[:, 3:], 3), 1: (b[:, 2:], 2)}
        )
        cache.begin_step(seq_ids)
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
        cache.begin_step([0] * 4)
        k, _, _ = cache.update_and_fetch(0, kv, kv, _positions(0, 4))
        self.assertEqual(k.shape[2], 4)  # four cells held, so a window of four

        cache.seq_rm(0)  # frees all four, so used_end walks back to 0
        self.assertEqual(cache.free_cells(), self.CAPACITY)

        # one token reclaims cell 0, so the window is its own single cell
        kv = torch.randn(1, self.n_kv_heads, 1, self.head_dim)
        cache.begin_step([1])
        k, _, spec = cache.update_and_fetch(0, kv, kv, torch.tensor([[0]]))
        self.assertEqual(k.shape[2], 1)  # the window length is 1, not the old 4
        self.assertEqual(spec.mask.shape[-1], 1)

    def test_seq_rm_over_a_range_frees_only_that_window(self):
        cache = self._cache()
        self._step(cache, torch.randn(1, 5, self.hidden), [0, 1, 2, 3, 4], [0] * 5)

        cache.seq_rm(0, 0, 2)  # sliding window: drop the oldest two
        self.assertEqual(cache.seq_len(0), 3)
        self.assertEqual(cache.free_cells(), self.CAPACITY - 3)

        cache.seq_rm(0, 4)  # backtrack: drop position 4 onwards
        self.assertEqual(cache.seq_len(0), 2)
        self.assertEqual(cache.free_cells(), self.CAPACITY - 2)

    def test_every_verb_range_checks_the_seq_id(self):
        # An id past the bitset would set a bit no int64 can hold, surfacing
        # much later as an overflow while building the mask.
        cache = self._cache()
        for call in (
            lambda: cache.begin_step([MAX_SEQS]),
            lambda: cache.seq_cp(0, MAX_SEQS),
            lambda: cache.seq_cp(MAX_SEQS, 0),
            lambda: cache.seq_rm(MAX_SEQS),
            lambda: cache.seq_len(MAX_SEQS),
            lambda: cache.seq_len(-1),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_admission_fails_before_the_forward(self):
        cache = self._cache(capacity=4)
        self.assertFalse(cache.can_extend(5))
        with self.assertRaises(RuntimeError):
            cache.begin_step([0] * 5)

    def test_step_protocol_is_enforced(self):
        cache = self._cache()
        kv = torch.randn(1, self.n_kv_heads, 1, self.head_dim)
        pos = torch.tensor([[0]])

        with self.assertRaises(ValueError):  # a step with no tokens
            cache.begin_step([])

        cache.begin_step([0, 0])  # declares two tokens, forward carries one
        with self.assertRaises(ValueError):
            cache.update_and_fetch(0, kv, kv, pos)
        with self.assertRaises(RuntimeError):  # the failed attempt still cleared it
            cache.update_and_fetch(0, kv, kv, torch.tensor([[0], [1]]))

        cache.begin_step([0])
        cache.update_and_fetch(0, kv, kv, pos)
        with self.assertRaises(RuntimeError):  # a second step, no begin_step
            cache.update_and_fetch(0, kv, kv, pos)

    def test_growth_keeps_cell_indices_and_bytes(self):
        # A grown pool must append rows only: a cell's index is its name, held
        # by the plan and by _pos/_seq, so renumbering or dropping rows would
        # move history without anything noticing.
        cache = self._cache()
        first = torch.randn(1, self.n_kv_heads, 2, self.head_dim)
        cache.begin_step([0, 0])
        k, _, _ = cache.update_and_fetch(0, first, first, torch.tensor([[0], [1]]))
        self.assertEqual(k.shape[2], 2)  # a short session reserves a short pool

        rest = torch.randn(1, self.n_kv_heads, 6, self.head_dim)
        cache.begin_step([0] * 6)
        k, v, _ = cache.update_and_fetch(
            0, rest, rest, torch.tensor([[p] for p in range(2, 8)])
        )
        self.assertEqual(k.shape[2], 8)
        torch.testing.assert_close(k[:, :, :2, :], first)  # cells 0,1 unmoved
        torch.testing.assert_close(v[:, :, 2:, :], rest)

    def test_sizings_agree(self):
        x = torch.randn(1, 5, self.hidden)
        out = [
            self._step(self._cache(sizing=s), x, list(range(5)), [0] * 5)
            for s in (CacheSizing.DYNAMIC, CacheSizing.STATIC)
        ]
        torch.testing.assert_close(out[0], out[1])

    def test_a_verb_does_not_hide_a_missing_begin_step(self):
        # A sequence verb drops the memoized plan, which must not be mistaken
        # for the start of a step -- that would silently reuse the previous
        # step's sequence assignment for the new tokens.
        cache = self._cache()
        kv = torch.randn(1, self.n_kv_heads, 2, self.head_dim)
        pos = torch.tensor([[0], [0]])
        cache.begin_step([0, 1])
        cache.update_and_fetch(0, kv, kv, pos)

        cache.seq_rm(2)  # any verb; a no-op here beyond dropping the plan
        with self.assertRaises(RuntimeError):  # layer 0 was already served
            cache.update_and_fetch(0, kv, kv, pos)
        with self.assertRaises(RuntimeError):  # and the declaration is spent
            cache.update_and_fetch(1, kv, kv, pos)


class ContiguousSpecTest(unittest.TestCase):
    # Which mask semantic the cache declares for each shape of step.

    def setUp(self):
        torch.manual_seed(0)
        self.cache = ContiguousReferenceCache(
            CacheConfig(n_layers=1, n_kv_heads=2, head_dim=4, capacity=8)
        )

    def _step(self, start, q_len):
        kv = torch.randn(1, 2, q_len, 4)
        return self.cache.update_and_fetch(0, kv, kv, _positions(start, q_len))[2]

    def test_decode_is_unmasked(self):
        self.assertEqual(self._step(0, 1).kind, MaskKind.NONE)

    def test_fresh_prefill_is_causal(self):
        self.assertEqual(self._step(0, 4).kind, MaskKind.CAUSAL)

    def test_continuation_is_an_explicit_lower_right_band(self):
        self._step(0, 4)
        spec = self._step(4, 3)  # 3 new cells at the tail of a 7-cell window
        self.assertEqual(spec.kind, MaskKind.EXPLICIT)
        self.assertEqual(spec.mask.dtype, torch.bool)
        torch.testing.assert_close(
            spec.mask, torch.ones(3, 7, dtype=torch.bool).tril(7 - 3)
        )


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

    def _attend(self, spec, k=None, v=None):
        k = self.k if k is None else k
        v = self.v if v is None else v
        return attend(self.q, k, v, spec, self.scale, torch.float32)

    def test_causal_rejects_a_non_square_window(self):
        # torch's is_causal is upper-left, so it cannot serve a continuation;
        # a cache must declare EXPLICIT there rather than CAUSAL.
        with self.assertRaises(ValueError):
            self._attend(AttendSpec(kind=MaskKind.CAUSAL))

    def test_explicit_attends_the_true_cells(self):
        # Polarity: masking to cells {0, 2} must equal attending over just those
        # keys. The inverted convention would instead select {1, 3, 4}.
        keep = torch.tensor([0, 2])
        mask = torch.zeros(self.q_len, self.total, dtype=torch.bool)
        mask[:, keep] = True
        torch.testing.assert_close(
            self._attend(AttendSpec(kind=MaskKind.EXPLICIT, mask=mask)),
            self._attend(
                AttendSpec(kind=MaskKind.NONE),
                self.k.index_select(2, keep),
                self.v.index_select(2, keep),
            ),
        )


if __name__ == "__main__":
    unittest.main()
