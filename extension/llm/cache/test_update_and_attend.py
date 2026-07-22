# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from executorch.extension.llm.cache.reference_cache import (
    CacheConfig,
    CacheSizing,
    ContiguousReferenceCache,
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


if __name__ == "__main__":
    unittest.main()
