# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import torch
import torch.nn.functional as F

from executorch.backends.cuda.triton.kernels.sdpa_varlen import sdpa_varlen
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    build_dflash_swa_mask,
)


def _reference(
    q, k_store, v_store, segments, scale, causal, window, ring, symmetric=False
):
    """Per-segment attention over logical positions, in float32."""
    _, hq, t_total, d = q.shape
    hkv = k_store.shape[2]
    out = torch.zeros_like(q, dtype=torch.float32)
    for q_start, q_len, col, kv_end in segments:
        if q_len == 0:
            continue
        positions = torch.arange(kv_end, device=q.device)
        rows = positions % ring if ring else positions
        keep = positions > kv_end - 1 - ring if ring else positions >= 0
        k = k_store[rows, col].float()  # [kv, H, D]
        v = v_store[rows, col].float()
        k = k.repeat_interleave(hq // hkv, dim=1).transpose(0, 1)
        v = v.repeat_interleave(hq // hkv, dim=1).transpose(0, 1)
        qs = q[0, :, q_start : q_start + q_len].float()
        scores = qs @ k.transpose(1, 2) * scale  # [H, q, kv]
        query_pos = torch.arange(kv_end - q_len, kv_end, device=q.device)
        visible = keep[None, :].expand(q_len, -1).clone()
        if causal:
            visible &= positions[None, :] <= query_pos[:, None]
        if window:
            visible &= positions[None, :] > query_pos[:, None] - window
            if symmetric:
                visible &= positions[None, :] < query_pos[:, None] + window
        scores = scores.masked_fill(~visible, float("-inf"))
        out[0, :, q_start : q_start + q_len] = torch.softmax(scores, -1) @ v
    return out


def _dflash_reference(q, k_store, v_store, segments, scale, window, ring):
    """Use the drafter's per-query mask over retained context and draft K/V."""
    out = torch.zeros_like(q, dtype=torch.float32)
    groups = q.shape[1] // k_store.shape[2]
    for start, block, col, end in segments:
        if block == 0:
            continue
        context = end - block
        slots = torch.arange(window, device=q.device)
        context_positions = slots + ((context - 1 - slots) // window) * window
        positions = torch.cat(
            (context_positions, torch.arange(context, end, device=q.device))
        ).clamp_min(0)
        rows = positions % ring if ring else positions
        keys = k_store[rows, col].float().repeat_interleave(groups, dim=1)
        values = v_store[rows, col].float().repeat_interleave(groups, dim=1)
        mask = build_dflash_swa_mask(context, block, window, window).to(q.device)
        out[:, :, start : start + block] = F.scaled_dot_product_attention(
            q[:, :, start : start + block].float(),
            keys.transpose(0, 1)[None],
            values.transpose(0, 1)[None],
            attn_mask=mask,
            scale=scale,
        )
    return out


def _dflash_inputs(
    contexts, block, window, ring, *, hq=8, hkv=2, d=64, visible_values=False
):
    rows = ring or max(contexts) + block
    q = torch.zeros(
        1, hq, len(contexts) * block, d, device="cuda", dtype=torch.bfloat16
    )
    q[..., 0] = 1
    k = torch.zeros(rows, 4, hkv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    segments = []
    for col, context in enumerate(contexts):
        oldest = max(0, context - window)
        row = oldest % ring if ring else oldest
        # Make dropping even the oldest retained context observable: ordinary
        # per-query windows omit this key from every draft block query.
        k[row, col, :, 0] = 100
        v[row, col] = 1
        if visible_values:
            values = (
                2 * (col + 1)
                + context / window
                + torch.arange(hkv, device=q.device)[:, None]
                + torch.arange(d, device=q.device)[None, :] / d
            )
            # The first key is visible only to query zero; the draft key stays
            # visible to every query. Keep the excluded boundary sentinel above.
            for position, contribution in (
                (context - window + 1, values + 8),
                (context + block - 1, values),
            ):
                row = position % ring if ring else position
                k[row, col, :, 0] = 96
                v[row, col] = contribution
        segments.append((col * block, block, col, context + block))
    padded = segments + [(0, 0, 0, 0)] * (4 - len(segments))
    info = torch.tensor(padded, device="cuda", dtype=torch.int32)
    return q, k, v, info, segments


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SdpaVarlenTest(unittest.TestCase):
    def _nonzero_dflash_reference(self, q, k, v, segments, scale, window, ring):
        want = _dflash_reference(q, k, v, segments, scale, window, ring)
        positions = _reference(
            q, k, v, segments, scale, False, window, ring, symmetric=True
        )
        torch.testing.assert_close(want, positions, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(want).all())
        self.assertGreater(want.abs().min().item(), 0.5)
        groups = q.shape[1] // k.shape[2]
        for start, block, _, _ in segments:
            self.assertGreater(
                (want[:, :, start] - want[:, :, start + 1]).abs().min().item(), 1
            )
            first_group = want[:, 0, start : start + block]
            next_group = want[:, groups, start : start + block]
            self.assertGreater((first_group - next_group).abs().min().item(), 0.5)
        for previous, current in zip(segments, segments[1:]):
            self.assertGreater(
                (want[:, :, previous[0]] - want[:, :, current[0]]).abs().min().item(),
                0.5,
            )
        return want

    def _run(
        self,
        segments,
        *,
        hq=8,
        hkv=2,
        d=64,
        rows=512,
        cols=5,
        t=None,
        causal=True,
        window=0,
        ring=0,
        seed=0,
    ):
        torch.manual_seed(seed)
        dev = "cuda"
        t = t or max(s + n for s, n, _, _ in segments)
        q = torch.randn(1, hq, t, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn(rows, cols, hkv, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn(rows, cols, hkv, d, device=dev, dtype=torch.bfloat16)
        info = torch.tensor(segments, dtype=torch.int32, device=dev)
        scale = d**-0.5
        got = sdpa_varlen(q, k, v, info, scale, causal, window, ring)
        want = _reference(q, k, v, segments, scale, causal, window, ring)
        torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)
        covered = torch.zeros(t, dtype=torch.bool)
        for s, n, _, _ in segments:
            covered[s : s + n] = True
        self.assertTrue((got[0, :, ~covered] == 0).all())

    def test_decode_rows_see_only_their_sequence(self):
        for kv in (1, 63, 64, 200, 511):
            self._run([(0, 1, 3, kv), (1, 1, 0, 37), (2, 1, 1, 512), (3, 1, 4, 5)])

    def test_prefill_and_decode_mix(self):
        self._run([(0, 100, 2, 150), (100, 1, 0, 300), (101, 30, 4, 30)])

    def test_padding_segments_and_rows(self):
        self._run([(0, 3, 1, 90), (0, 0, 0, 0), (3, 2, 2, 7)], t=16)

    def test_one_owner_split_across_segments(self):
        # The second segment's rows see the first's, written in the same step.
        self._run([(0, 4, 2, 44), (4, 3, 1, 10), (7, 5, 2, 49)])

    def test_sliding_window(self):
        for window in (1, 16, 100):
            self._run([(0, 64, 0, 200), (64, 1, 3, 300)], window=window)

    def test_ring(self):
        ring = 96
        for kv_end in (40, 96, 97, 300):
            self._run(
                [(0, 8, 1, kv_end), (8, 1, 2, max(kv_end - 3, 1))],
                window=64,
                ring=ring,
                rows=ring,
            )

    def test_non_causal_draft_block(self):
        self._run([(0, 16, 0, 80), (16, 16, 3, 16)], causal=False)

    def test_dflash_context_window_matches_original_mask(self):
        window = 2048
        for block, ring, visible_values in itertools.product(
            (2, 16), (0, 2175), (False, True)
        ):
            q, k, v, info, segments = _dflash_inputs(
                (3000, 7100), block, window, ring, visible_values=visible_values
            )
            got = sdpa_varlen(
                q, k, v, info, 0.125, False, window, ring, symmetric_window=True
            )
            if visible_values:
                want = self._nonzero_dflash_reference(
                    q, k, v, segments, 0.125, window, ring
                )
            else:
                want = _dflash_reference(q, k, v, segments, 0.125, window, ring)
                self.assertTrue((want == 0).all())
            torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)
            ordinary = sdpa_varlen(q, k, v, info, 0.125, False, window, ring)
            ordinary_want = _reference(q, k, v, segments, 0.125, False, window, ring)
            torch.testing.assert_close(
                ordinary.float(), ordinary_want, atol=2e-2, rtol=2e-2
            )
            shared = sdpa_varlen(
                q, k, v, info, 0.125, False, window, ring, context_window=True
            )
            self.assertGreater((shared.float() - got.float()).abs().max().item(), 0.9)

    def test_dflash_context_window_muse_head_geometry(self):
        window = 2048
        for block in (2, 16):
            ring = window + block
            q, k, v, info, segments = _dflash_inputs(
                (3000, 7100),
                block,
                window,
                ring,
                hq=32,
                hkv=8,
                d=128,
                visible_values=True,
            )
            scale = 128**-0.5
            got = sdpa_varlen(
                q, k, v, info, scale, False, window, ring, symmetric_window=True
            )
            want = self._nonzero_dflash_reference(
                q, k, v, segments, scale, window, ring
            )
            torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)

    def test_dflash_context_window_graph_replay(self):
        window, ring, block = 2048, 2175, 16
        q, k, v, info, _ = _dflash_inputs(
            (4096, 6000), block, window, ring, visible_values=True
        )
        sdpa_varlen(q, k, v, info, 0.125, False, window, ring, symmetric_window=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = sdpa_varlen(
                q, k, v, info, 0.125, False, window, ring, symmetric_window=True
            )
        previous_want = None
        for contexts in ((4096, 6000), (8192, 2500)):
            _, new_k, new_v, new_info, segments = _dflash_inputs(
                contexts, block, window, ring, visible_values=True
            )
            k.copy_(new_k)
            v.copy_(new_v)
            info.copy_(new_info)
            graph.replay()
            want = self._nonzero_dflash_reference(
                q, k, v, segments, 0.125, window, ring
            )
            if previous_want is not None:
                self.assertGreater((want - previous_want).abs().min().item(), 0.5)
            torch.testing.assert_close(out.float(), want, atol=2e-2, rtol=2e-2)
            previous_want = want.clone()

    def test_dynamic_dflash_context_window(self):
        window, ring, block = 2048, 2175, 16
        compiled = torch.compile(
            lambda q, k, v, info: sdpa_varlen(
                q, k, v, info, 0.125, False, window, ring, symmetric_window=True
            ),
            fullgraph=True,
            dynamic=True,
        )
        torch._dynamo.reset()
        frames = torch._dynamo.utils.counters["stats"]
        before = frames["unique_graphs"]
        for owners in (2, 3, 4):
            q, k, v, info, segments = _dflash_inputs(
                (3000, 5000, 7000, 9000)[:owners],
                block,
                window,
                ring,
                visible_values=True,
            )
            got = compiled(q, k, v, info)
            want = self._nonzero_dflash_reference(
                q, k, v, segments, 0.125, window, ring
            )
            torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)
        self.assertEqual(frames["unique_graphs"] - before, 1)

    def test_dflash_symmetric_windows_match_position_and_model_oracles(self):
        for window, block in itertools.product((1, 3, 8), (1, 2, 4, 16)):
            for ring in (0, window + block - 1, 512):
                with self.subTest(window=window, block=block, ring=ring):
                    q, k, v, info, segments = _dflash_inputs(
                        (0, window - 1, window + 1, 3 * window), block, window, ring
                    )
                    k.zero_()
                    slots = torch.arange(k.shape[0], device=q.device)
                    for _, _, col, end in segments:
                        positions = (
                            slots + ring * ((end - 1 - slots) // ring)
                            if ring
                            else slots
                        )
                        v[:, col] = positions[:, None, None] / 16
                    got = sdpa_varlen(
                        q, k, v, info, 0.125, False, window, ring, symmetric_window=True
                    )
                    want = _reference(
                        q, k, v, segments, 0.125, False, window, ring, symmetric=True
                    )
                    model = _dflash_reference(q, k, v, segments, 0.125, window, ring)
                    torch.testing.assert_close(model, want, atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)

    def test_dflash_strict_right_boundary(self):
        window, block, context = 3, 8, 20
        for ring in (0, window + block - 1):
            q, k, v, info, segments = _dflash_inputs((context,), block, window, ring)
            k.zero_()
            v.zero_()
            row = (context + window) % ring if ring else context + window
            k[row, 0, :, 0] = 100
            v[row, 0] = 1
            got = sdpa_varlen(
                q, k, v, info, 0.125, False, window, ring, symmetric_window=True
            )
            want = _dflash_reference(q, k, v, segments, 0.125, window, ring)
            torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)
            self.assertTrue((got[:, :, 0] == 0).all())
            self.assertTrue((got[:, :, 1:6] > 0.9).all())
            self.assertTrue((got[:, :, 6:] == 0).all())
            ordinary = sdpa_varlen(q, k, v, info, 0.125, False, window, ring)
            self.assertTrue((ordinary[:, :, 0] > 0.9).all())

    def test_dflash_zero_window_remains_full_attention(self):
        self._run([(0, 2, 0, 80)], causal=False)
        q, k, v, info, segments = _dflash_inputs((20,), 2, 8, 0)
        got = sdpa_varlen(q, k, v, info, 0.125, False, symmetric_window=True)
        want = _reference(q, k, v, segments, 0.125, False, 0, 0)
        torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)

    def test_dflash_graph_replay_reads_new_query_lengths(self):
        window, ring = 3, 512
        q, k, v, info, _ = _dflash_inputs((20,), 16, window, ring)
        k.zero_()
        v[:, 0] = torch.arange(ring, device=q.device)[:, None, None] / 32
        sdpa_varlen(q, k, v, info, 0.125, False, window, ring, symmetric_window=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = sdpa_varlen(
                q, k, v, info, 0.125, False, window, ring, symmetric_window=True
            )
        for length, end in ((1, 1), (2, 513), (8, 1030), (16, 900)):
            segments = [(0, length, 0, end)]
            info.copy_(torch.tensor(segments + [(0, 0, 0, 0)] * 3, dtype=torch.int32))
            graph.replay()
            want = _reference(
                q, k, v, segments, 0.125, False, window, ring, symmetric=True
            )
            model = _dflash_reference(q, k, v, segments, 0.125, window, ring)
            torch.testing.assert_close(model, want, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(out.float(), want, atol=2e-2, rtol=2e-2)
            self.assertTrue((out[:, :, length:] == 0).all())

    def test_head_layouts(self):
        for hq, hkv, d in itertools.product((4, 16), (1, 4), (64, 128)):
            if hq % hkv:
                continue
            self._run([(0, 2, 0, 130), (2, 9, 1, 9)], hq=hq, hkv=hkv, d=d)

    def test_graph_replay_reads_new_lengths(self):
        torch.manual_seed(1)
        dev = "cuda"
        q = torch.randn(1, 8, 4, 64, device=dev, dtype=torch.bfloat16)
        k = torch.randn(256, 3, 2, 64, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        info = torch.tensor(
            [[0, 1, 0, 10], [1, 1, 1, 20], [2, 2, 2, 30]], dtype=torch.int32, device=dev
        )
        sdpa_varlen(q, k, v, info, 0.125)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = sdpa_varlen(q, k, v, info, 0.125)
        for segments in (
            [[0, 1, 0, 200], [1, 1, 1, 3], [2, 2, 2, 256]],
            [[0, 4, 1, 100], [0, 0, 0, 0], [0, 0, 0, 0]],
        ):
            info.copy_(torch.tensor(segments, dtype=torch.int32))
            graph.replay()
            want = _reference(q, k, v, segments, 0.125, True, 0, 0)
            torch.testing.assert_close(out.float(), want, atol=2e-2, rtol=2e-2)

    def test_small_t_split_path(self):
        production = [(0, 1, 0, 600), (1, 1, 1, 900), (2, 1, 2, 2500), (3, 1, 3, 4096)]
        for window in (0, 2048):
            self._run(
                production,
                hq=32,
                hkv=2,
                d=128,
                rows=4096,
                cols=4,
                t=4,
                window=window,
            )
        self._run(
            [(0, 3, 1, 130), (3, 1, 0, 4096), (0, 0, 2, 0)],
            hq=32,
            hkv=2,
            d=128,
            rows=4096,
            t=4,
            causal=False,
        )
        self._run(
            [(0, 1, 1, 600), (2, 2, 0, 4096), (0, 0, 2, 0)],
            hq=32,
            hkv=2,
            d=128,
            rows=4096,
            t=4,
        )

    def test_small_t_ring_and_uncovered_rows(self):
        self._run(
            [(0, 1, 1, 300), (0, 0, 0, 0), (2, 2, 2, 97)],
            hq=32,
            hkv=2,
            d=128,
            rows=96,
            t=4,
            window=64,
            ring=96,
        )

    def test_overprovisioned_store_graph_replay(self):
        torch.manual_seed(2)
        dev = "cuda"
        q = torch.randn(1, 8, 4, 64, device=dev, dtype=torch.bfloat16)
        k = torch.randn(8192, 4, 2, 64, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        info = torch.tensor(
            [[0, 1, 0, 100], [1, 1, 1, 600], [2, 1, 2, 900], [3, 1, 3, 1000]],
            dtype=torch.int32,
            device=dev,
        )
        sdpa_varlen(q, k, v, info, 0.125)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = sdpa_varlen(q, k, v, info, 0.125)
        scenarios = (
            [[0, 1, 0, 100], [1, 1, 1, 600], [2, 1, 2, 900], [3, 1, 3, 1000]],
            [[0, 1, 3, 4096], [1, 1, 2, 17], [2, 1, 1, 8000], [3, 1, 0, 700]],
        )
        for segments in scenarios:
            info.copy_(torch.tensor(segments, dtype=torch.int32))
            graph.replay()
            want = _reference(q, k, v, segments, 0.125, True, 0, 0)
            torch.testing.assert_close(out.float(), want, atol=2e-2, rtol=2e-2)

    def test_rows_available(self):
        torch.manual_seed(3)
        dev = "cuda"
        hq, hkv, d, available = 8, 2, 64, 128
        q = torch.randn(1, hq, 1, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn(512, 3, hkv, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        info = torch.tensor([[0, 1, 2, 300]], dtype=torch.int32, device=dev)
        got = sdpa_varlen(q, k, v, info, d**-0.5, rows_available=available)

        keys = k[:available, 2].float().repeat_interleave(hq // hkv, dim=1)
        values = v[:available, 2].float().repeat_interleave(hq // hkv, dim=1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)
        scores = q[0, :, 0].float()[:, None, :] @ keys.transpose(1, 2)
        want = torch.softmax(scores * (d**-0.5), -1) @ values
        torch.testing.assert_close(
            got[0, :, 0].float(), want[:, 0], atol=2e-2, rtol=2e-2
        )

    def test_torch_compile(self):
        torch.manual_seed(4)
        dev = "cuda"
        q = torch.randn(1, 8, 4, 64, device=dev, dtype=torch.bfloat16)
        k = torch.randn(512, 2, 2, 64, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        segments = [(0, 1, 0, 300), (1, 3, 1, 500)]
        info = torch.tensor(segments, dtype=torch.int32, device=dev)

        def fn(q, k, v, info):
            return sdpa_varlen(q, k, v, info, 0.125, True, 256, 0)

        got = torch.compile(fn, fullgraph=True)(q, k, v, info)
        want = _reference(q, k, v, segments, 0.125, True, 256, 0)
        torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)

    def test_torch_compile_generic(self):
        torch.manual_seed(5)
        dev = "cuda"
        q = torch.randn(1, 8, 16, 64, device=dev, dtype=torch.bfloat16)
        k = torch.randn(512, 2, 2, 64, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        segments = [(0, 16, 0, 500), (0, 0, 1, 0)]
        info = torch.tensor(segments, dtype=torch.int32, device=dev)

        def fn(q, k, v, info):
            return sdpa_varlen(q, k, v, info, 0.125, True, 256, 0)

        got = torch.compile(fn, fullgraph=True)(q, k, v, info)
        want = _reference(q, k, v, segments, 0.125, True, 256, 0)
        torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)

    def test_dynamic_query_length_is_not_specialized(self):
        torch.manual_seed(6)
        dev = "cuda"
        k = torch.randn(512, 3, 2, 64, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        compiled = torch.compile(
            lambda q, k, v, info: sdpa_varlen(q, k, v, info, 0.125, False),
            fullgraph=True,
            dynamic=True,
        )
        torch._dynamo.reset()
        frames = torch._dynamo.utils.counters["stats"]
        before = frames["unique_graphs"]
        for owners in (2, 3, 4):
            t = owners * 16
            q = torch.randn(1, 8, t, 64, device=dev, dtype=torch.bfloat16)
            segments = [(16 * i, 16, i % 3, 40 + 16 * i) for i in range(owners)]
            # The segment table is fixed-size in a lowered step; pad it.
            padded = segments + [(0, 0, 0, 0)] * (4 - owners)
            info = torch.tensor(padded, dtype=torch.int32, device=dev)
            got = compiled(q, k, v, info)
            want = _reference(q, k, v, segments, 0.125, False, 0, 0)
            torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)
        self.assertEqual(frames["unique_graphs"] - before, 1)

    def test_rejects_unsupported_layouts(self):
        dev = "cuda"
        q = torch.randn(1, 8, 1, 64, device=dev, dtype=torch.bfloat16)
        k = torch.randn(128, 2, 2, 64, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        info = torch.zeros((1, 8), device=dev, dtype=torch.int32)[:, ::2]
        with self.assertRaisesRegex(ValueError, "seq_info must be contiguous"):
            sdpa_varlen(q, k, v, info, 0.125)

        q = torch.randn(1, 8, 1, 96, device=dev, dtype=torch.bfloat16)
        k = torch.randn(128, 2, 2, 96, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        info = torch.tensor([[0, 1, 0, 100]], dtype=torch.int32, device=dev)
        with self.assertRaisesRegex(ValueError, "head dimension must be a power"):
            sdpa_varlen(q, k, v, info, 96**-0.5)

        q = torch.randn(1, 8, 1, 64, device=dev, dtype=torch.bfloat16)
        k = torch.randn(128, 2, 2, 64, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        with self.assertRaisesRegex(ValueError, "context_window requires non-causal"):
            sdpa_varlen(q, k, v, info, 0.125, context_window=True)
        with self.assertRaisesRegex(ValueError, "symmetric_window requires"):
            sdpa_varlen(q, k, v, info, 0.125, symmetric_window=True)
        with self.assertRaisesRegex(ValueError, "symmetric_window requires"):
            sdpa_varlen(
                q,
                k,
                v,
                info,
                0.125,
                False,
                context_window=True,
                symmetric_window=True,
            )


if __name__ == "__main__":
    unittest.main()
