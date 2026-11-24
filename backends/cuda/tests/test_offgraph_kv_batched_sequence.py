# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The batched-sequence off-graph KV layout on CUDA, against the neutral op.

The oracle is ``BatchedSequenceReferenceCache``: every sequence keeps a private
history, and a packed step's tokens attend only their own sequence -- the same
contract MLX's batched-sequence cache implements. The lowered decomposition
reads the step from the two buffers the runtime cache writes: each token's
storage column (``seq_cols``) and each run of one sequence's rows
(``seq_info``).
"""

import json
import unittest

import torch
import torch.nn.functional as F
from executorch.backends.cuda.passes.lower_offgraph_kv import (
    batched_sequence_step,
    CheckOffGraphKVStepWidthPass,
    LowerOffGraphKVPass,
    offgraph_kv_constant_methods,
    OFFGRAPH_KV_FQN_PREFIX,
    OFFGRAPH_KV_SEQ_COLS_FQN,
    OFFGRAPH_KV_SEQ_INFO_FQN,
    parse_offgraph_kv_manifest,
    ring_physical_capacity,
)
from executorch.backends.cuda.triton.kernels.sdpa_varlen import _num_splits, sdpa_varlen
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    build_dflash_swa_mask,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.llm.cache.reference_cache import (
    BatchedSequenceReferenceCache,
    CacheConfig,
    LayerPolicy,
)
from executorch.extension.llm.cache.update_and_attend import REGISTRY

N_KV_HEADS = 2
N_Q_HEADS = 4
HEAD_DIM = 64
SCALE = 0.125


def _inputs(start: int, length: int):
    q = torch.randn(1, N_Q_HEADS, length, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        1, N_KV_HEADS, length, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    return q, k, v, torch.arange(start, start + length, device="cuda")


def _batch(groups):
    """Packs (seq_id, start, length) groups onto one token axis."""
    parts = [_inputs(start, length) for _, start, length in groups]
    seq_ids = [seq for seq, _, length in groups for _ in range(length)]
    q, k, v = (torch.cat([p[i] for p in parts], dim=2) for i in range(3))
    position = torch.cat([p[3] for p in parts])
    return seq_ids, q, k, v, position


def _plan(seq_ids, position, columns, width, max_segments, scratch):
    """What the runtime cache writes before a forward: each token's column,
    and one (q_start, q_len, col, kv_end, physical_columns) row per run of one sequence.
    """
    seq_cols = [columns[s] for s in seq_ids] + [scratch] * (width - len(seq_ids))
    info, start = [], 0
    positions = position.tolist()
    while start < len(seq_ids):
        end = start
        while end < len(seq_ids) and seq_ids[end] == seq_ids[start]:
            end += 1
        info.append(
            (
                start,
                end - start,
                columns[seq_ids[start]],
                positions[end - 1] + 1,
                scratch + 1,
            )
        )
        start = end
    info += [(0, 0, 0, 0, scratch + 1)] * (max_segments - len(info))
    return (
        torch.tensor(seq_cols, dtype=torch.long, device="cuda"),
        torch.tensor(info, dtype=torch.int32, device="cuda"),
    )


class _Oracle:
    def __init__(self, capacity: int, windows) -> None:
        self.windows = tuple(windows)
        self._cache = BatchedSequenceReferenceCache(
            CacheConfig(
                n_layers=len(self.windows),
                n_kv_heads=N_KV_HEADS,
                head_dim=HEAD_DIM,
                capacity=capacity,
                dtype=torch.float32,
                layers=tuple(
                    LayerPolicy.ring(w) if w else LayerPolicy.flat()
                    for w in self.windows
                ),
            )
        )
        self._key = f"cuda-offgraph-batched-sequence-{id(self)}"
        REGISTRY.install(self._key, self._cache)

    def step(self, seq_ids, q, k, v, position):
        self._cache.declare_step(seq_ids)
        outputs = []
        with REGISTRY.active(self._key):
            for layer_id in range(len(self.windows)):
                outputs.append(
                    torch.ops.kvcache.update_and_attend(
                        q.float().cpu(),
                        k.float().cpu(),
                        v.float().cpu(),
                        position.reshape(-1, 1).cpu(),
                        layer_id,
                        SCALE,
                        torch.float32,
                    )
                )
        return outputs

    def close(self) -> None:
        REGISTRY.uninstall(self._key)


def _max_abs_diff(actual, expected) -> float:
    return (actual.float().cpu() - expected.float().cpu()).abs().max().item()


# Two sequences prefilling in one step, decoding together, then one decoding
# beside the other's chunk, then a third joining late.
_STEPS = (
    ((0, 0, 5), (1, 0, 3)),
    ((0, 5, 1), (1, 3, 1)),
    ((1, 4, 4), (0, 6, 1)),
    ((2, 0, 6), (0, 7, 1), (1, 8, 1)),
    ((0, 8, 1), (2, 6, 1)),
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BatchedSequenceStepTest(unittest.TestCase):
    """batched_sequence_step, fed the runtime's plan, matches the neutral op."""

    MAX_SEQS = 3
    MAX_WRITE = 16

    def _run(self, steps, capacity, window=0, max_segments=6, initial_rows=0):
        torch.manual_seed(20)
        oracle = _Oracle(4 * capacity, (window,))
        self.addCleanup(oracle.close)
        columns = {seq: seq for seq in range(self.MAX_SEQS)}
        cols = self.MAX_SEQS + 1
        ring = window > 0
        rows = (
            min(capacity, ring_physical_capacity(window, self.MAX_WRITE))
            if ring
            else capacity
        )
        if ring and initial_rows:
            raise ValueError("initial_rows only exercises flat growth")
        live_rows = initial_rows or rows
        k_store = torch.zeros(
            1,
            live_rows * cols,
            N_KV_HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        v_store = torch.zeros_like(k_store)
        for groups in steps:
            seq_ids, q, k, v, position = _batch(groups)
            (expected,) = oracle.step(seq_ids, q, k, v, position)
            if int(position.max()) + 1 > live_rows:
                grown_k = torch.zeros(
                    1,
                    rows * cols,
                    N_KV_HEADS,
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                grown_k.view(rows, cols, N_KV_HEADS, HEAD_DIM)[:live_rows].copy_(
                    k_store.view(live_rows, cols, N_KV_HEADS, HEAD_DIM)
                )
                grown_v = torch.zeros_like(grown_k)
                grown_v.view(rows, cols, N_KV_HEADS, HEAD_DIM)[:live_rows].copy_(
                    v_store.view(live_rows, cols, N_KV_HEADS, HEAD_DIM)
                )
                k_store, v_store, live_rows = grown_k, grown_v, rows
            # A static method one wider than the step: one padding row.
            width = len(seq_ids) + 1
            pad = torch.zeros(
                1, q.shape[1], 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16
            )
            kpad = torch.zeros(
                1, N_KV_HEADS, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16
            )
            seq_cols, seq_info = _plan(
                seq_ids, position, columns, self.MAX_WRITE, max_segments, cols - 1
            )
            out = batched_sequence_step(
                torch.cat([q, pad], 2),
                torch.cat([k, kpad], 2),
                torch.cat([v, kpad], 2),
                torch.cat([position, position.new_zeros(1)]),
                k_store,
                v_store,
                seq_cols,
                seq_info,
                SCALE,
                live_rows,
                cols,
                window,
                ring,
                True,
            )
            self.assertLess(_max_abs_diff(out[:, :, : width - 1], expected), 2e-2)
            self.assertTrue((out[:, :, width - 1 :] == 0).all())
        return k_store, rows, cols

    def test_sequences_see_only_their_own_history(self) -> None:
        self._run(_STEPS, capacity=32)

    def test_windowed_ring_layer(self) -> None:
        self._run(_STEPS, capacity=32, window=4)

    def test_ring_wrap_matches_the_oracle(self) -> None:
        # Positions past the ring rows (19) wrap; the oracle stays authoritative.
        self._run(
            (((0, 0, 8),), ((0, 8, 8),), ((0, 16, 8),), ((0, 24, 2),)),
            capacity=32,
            window=4,
        )

    def test_flat_growth_matches_the_oracle(self) -> None:
        self._run(
            (((0, 0, 8),), ((0, 8, 8),), ((0, 16, 8),)),
            capacity=32,
            initial_rows=8,
        )

    def test_padding_writes_only_the_scratch_column(self) -> None:
        k_store, rows, cols = self._run(_STEPS[:1], capacity=16)
        grid = k_store.view(rows, cols, N_KV_HEADS, HEAD_DIM)
        # Sequence 2 never ran; only row 0 of the scratch column took padding.
        self.assertFalse(grid[:, 2].any())
        self.assertFalse(grid[1:, cols - 1].any())


class _FlatAndRing(torch.nn.Module):
    def forward(self, q, k, v, position):
        flat = torch.ops.kvcache.update_and_attend(
            q, k, v, position, 0, SCALE, torch.bfloat16
        )
        ring = torch.ops.kvcache.update_and_attend(
            q, k, v, position, 1, SCALE, torch.bfloat16
        )
        return flat + ring


class _DraftRing(torch.nn.Module):
    def forward(self, q, k, v, position):
        return torch.ops.kvcache.update_and_attend(
            q, k, v, position, 0, SCALE, torch.bfloat16
        )


def _draft_model_reference(q, storage, segments, window, rows):
    out = torch.zeros_like(q, dtype=torch.float32)
    for start, block, col, end in segments:
        context = end - block
        slots = torch.arange(window, device=q.device)
        context_positions = slots + ((context - 1 - slots) // window) * window
        positions = torch.cat(
            (context_positions, torch.arange(context, end, device=q.device))
        )
        values = (
            storage[positions.clamp_min(0) % rows, col].transpose(0, 1)[None].float()
        )
        mask = build_dflash_swa_mask(context, block, window, window).to(q.device)
        out[:, :, start : start + block] = F.scaled_dot_product_attention(
            q[:, :, start : start + block].float(),
            torch.zeros_like(values),
            values,
            attn_mask=mask,
            scale=SCALE,
            enable_gqa=True,
        )
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RuntimeColumnPitchTest(unittest.TestCase):
    """Check runtime write/read pitch with unchanged AOT tensor dimensions.

    This test reserves the declared maximum tensor and poisons the unused
    tail. Actual smaller CUDA allocations are tested by the C++ cache and
    model probe; this numerical test does not establish memory savings.
    """

    def _run(self, window):
        torch.manual_seed(125)
        declared_columns, max_write = 22, 32
        rows = min(64, window + max_write - 1) if window else 64
        oracle = _Oracle(128, (window,))
        self.addCleanup(oracle.close)
        k_store = torch.full(
            (1, rows * declared_columns, N_KV_HEADS, HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda",
        )
        v_store = torch.full_like(k_store, float("nan"))
        old_mapping, old_columns = {}, 0
        positions = {}
        for active in (1, 2, 4, 8, 16, 1):
            owners = list(range(active)) if active != 1 or not old_mapping else [5]
            mapping = {owner: column for column, owner in enumerate(owners)}
            physical_columns = len(owners) + 1
            for store in (k_store, v_store):
                retained = {
                    owner: store[0, : rows * old_columns]
                    .view(rows, old_columns, N_KV_HEADS, HEAD_DIM)[:, column]
                    .clone()
                    for owner, column in old_mapping.items()
                    if owner in mapping
                }
                store.fill_(float("nan"))
                compact = store[0, : rows * physical_columns].view(
                    rows, physical_columns, N_KV_HEADS, HEAD_DIM
                )
                compact.zero_()
                for owner, values in retained.items():
                    compact[:, mapping[owner]].copy_(values)
            groups = []
            for owner in owners:
                start = positions.get(owner, 0)
                # Different histories exercise segment boundaries and ring wrap.
                length = 1 + owner % 2 if active < 16 else 1
                if not old_mapping:
                    length = max_write - 1
                groups.append((owner, start, length))
                positions[owner] = start + length
            seq_ids, q, k, v, position = _batch(groups)
            (expected,) = oracle.step(seq_ids, q, k, v, position)
            seq_cols, seq_info = _plan(
                seq_ids,
                position,
                mapping,
                max_write,
                42,
                physical_columns - 1,
            )
            # Exercise an uncovered padded row in both split and generic paths.
            padded_q = torch.cat([q, torch.zeros_like(q[:, :, :1])], dim=2)
            padded_k = torch.cat([k, torch.zeros_like(k[:, :, :1])], dim=2)
            padded_v = torch.cat([v, torch.zeros_like(v[:, :, :1])], dim=2)
            out = batched_sequence_step(
                padded_q,
                padded_k,
                padded_v,
                torch.cat([position, position.new_zeros(1)]),
                k_store,
                v_store,
                seq_cols,
                seq_info,
                SCALE,
                rows,
                declared_columns,
                window,
                window > 0,
                True,
            )
            self.assertLess(_max_abs_diff(out[:, :, : len(seq_ids)], expected), 2e-2)
            self.assertTrue((out[:, :, len(seq_ids) :] == 0).all())
            for store in (k_store, v_store):
                self.assertTrue(torch.isnan(store[0, rows * physical_columns :]).all())
            old_mapping, old_columns = mapping, physical_columns

    def test_flat_pitch_and_sparse_survivor(self):
        self._run(window=0)

    def test_ring_pitch_and_sparse_survivor(self):
        self._run(window=2)

    def test_runtime_pitch_rejects_noncontiguous_column_and_head_lanes(self):
        query, _, _, _ = _inputs(0, 1)
        key = torch.zeros(
            (8, 3, N_KV_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
        )
        info = torch.tensor([[0, 1, 0, 1, 3]], device="cuda", dtype=torch.int32)
        layouts = [
            key.transpose(1, 2).contiguous().transpose(1, 2),
            key.repeat_interleave(2, dim=1)[:, ::2],
        ]
        for bad in layouts:
            for bad_key in (False, True):
                with self.subTest(stride=bad.stride(), bad_key=bad_key):
                    with self.assertRaisesRegex(ValueError, "column and head lanes"):
                        sdpa_varlen(
                            query,
                            bad if bad_key else key,
                            key if bad_key else bad,
                            info,
                            SCALE,
                        )

    def test_runtime_pitch_with_multiple_decode_splits(self):
        torch.manual_seed(126)
        rows, declared_columns, physical_columns = 1024, 9, 4
        query = torch.randn(
            1, N_Q_HEADS, 3, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        key = torch.full(
            (rows, declared_columns, N_KV_HEADS, HEAD_DIM),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.full_like(key, float("nan"))
        live_key = key.reshape(-1, N_KV_HEADS, HEAD_DIM)[
            : rows * physical_columns
        ].view(rows, physical_columns, N_KV_HEADS, HEAD_DIM)
        live_value = value.reshape(-1, N_KV_HEADS, HEAD_DIM)[
            : rows * physical_columns
        ].view_as(live_key)
        live_key.normal_()
        live_value.normal_()
        info = torch.tensor(
            [
                [0, 1, 2, rows, physical_columns],
                [1, 1, 0, 768, physical_columns],
                [0, 0, 0, 0, physical_columns],
            ],
            dtype=torch.int32,
            device="cuda",
        )
        self.assertGreater(_num_splits(query.shape[2], rows), 1)
        actual = sdpa_varlen(query, key, value, info, SCALE, rows_available=rows)
        expected = torch.zeros_like(query)
        for token, column, length in ((0, 2, rows), (1, 0, 768)):
            k = (
                live_key[:length, column]
                .transpose(0, 1)
                .repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=0)
                .float()
            )
            v = (
                live_value[:length, column]
                .transpose(0, 1)
                .repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=0)
                .float()
            )
            scores = (query[0, :, token, None].float() * k).sum(-1) * SCALE
            expected[0, :, token] = (
                (scores.softmax(-1)[..., None] * v).sum(1).to(expected.dtype)
            )
        self.assertLess(_max_abs_diff(actual, expected), 2e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LowerBatchedSequencePassTest(unittest.TestCase):
    CAPACITY = 32
    MAX_WRITE = 8
    MAX_SEQS = 3
    MAX_SEGMENTS = 6
    WINDOW = 4

    @classmethod
    def manifest(cls, **overrides) -> bytes:
        manifest = {
            "version": 1,
            "layout": "batched-sequence",
            "maximum_capacity": cls.CAPACITY,
            "max_write": cls.MAX_WRITE,
            "max_seqs": cls.MAX_SEQS,
            "max_segments": cls.MAX_SEGMENTS,
            "layers": [
                {"layer_id": 0, "policy": "flat"},
                {"layer_id": 1, "policy": "ring", "window": cls.WINDOW},
            ],
        }
        manifest.update(overrides)
        return json.dumps(manifest).encode()

    def _lowered(self, manifest: bytes | None = None):
        q, k, v, position = _inputs(0, self.MAX_WRITE)
        t = torch.export.Dim("t", min=1, max=self.MAX_WRITE)
        program = torch.export.export(
            _FlatAndRing(),
            (q, k, v, position.reshape(-1, 1)),
            dynamic_shapes=({2: t}, {2: t}, {2: t}, {0: t}),
            strict=False,
        )
        edge = to_edge(
            program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        ).exported_program()
        if manifest is None:
            manifest = self.manifest()
        return LowerOffGraphKVPass(parse_offgraph_kv_manifest(manifest))(edge)

    def test_manifest_is_validated(self) -> None:
        parsed = parse_offgraph_kv_manifest(self.manifest())
        self.assertEqual(parsed["layout"], "batched-sequence")
        self.assertTrue(parsed["causal"])
        self.assertFalse(parsed["context_window"])
        self.assertFalse(parsed["symmetric_window"])
        self.assertFalse(
            parse_offgraph_kv_manifest(self.manifest(causal=False))["causal"]
        )
        for bad in (
            {"max_seqs": 0},
            {"max_segments": None},
            {"max_segments": 5},
            {"causal": 1},
            {"context_window": 1},
            {"context_window": True},
            {"symmetric_window": 1},
            {"symmetric_window": True},
            {"causal": False, "context_window": True, "symmetric_window": True},
            {"max_write": self.CAPACITY + 1},
        ):
            with self.assertRaises(ValueError):
                parse_offgraph_kv_manifest(self.manifest(**bad))

    def test_physical_pitch_is_the_default_interface(self) -> None:
        manifest = self.manifest()
        lowered = self._lowered(manifest)
        buffers = {
            n.name: tuple(n.meta["val"].shape)
            for n in lowered.graph_module.graph.nodes
            if n.op == "placeholder" and n.name.startswith(OFFGRAPH_KV_FQN_PREFIX)
        }
        self.assertEqual(buffers[OFFGRAPH_KV_SEQ_INFO_FQN], (self.MAX_SEGMENTS, 5))
        self.assertEqual(
            offgraph_kv_constant_methods(manifest),
            {"get_offgraph_kv_max_seqs": self.MAX_SEQS},
        )
        with self.assertRaises(ValueError):
            parse_offgraph_kv_manifest(self.manifest(version=2))

    def test_declares_column_storage_and_step_buffers(self) -> None:
        lowered = self._lowered()
        graph = lowered.graph_module.graph
        target = exir_ops.edge.kvcache.update_and_attend.default
        self.assertFalse(any(n.target == target for n in graph.nodes))
        buffers = {
            n.name: tuple(n.meta["val"].shape)
            for n in graph.nodes
            if n.op == "placeholder" and n.name.startswith(OFFGRAPH_KV_FQN_PREFIX)
        }
        cols = self.MAX_SEQS + 1
        ring = min(self.CAPACITY, ring_physical_capacity(self.WINDOW, self.MAX_WRITE))
        flat = (1, self.CAPACITY * cols, N_KV_HEADS, HEAD_DIM)
        windowed = (1, ring * cols, N_KV_HEADS, HEAD_DIM)
        self.assertEqual(
            buffers,
            {
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_0_k": flat,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_0_v": flat,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_1_k": windowed,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_1_v": windowed,
                OFFGRAPH_KV_SEQ_COLS_FQN: (self.MAX_WRITE,),
                OFFGRAPH_KV_SEQ_INFO_FQN: (self.MAX_SEGMENTS, 5),
            },
        )
        calls = [
            n for n in graph.nodes if n.target == torch.ops.triton.sdpa_varlen.default
        ]
        self.assertEqual(len(calls), 2)

    def _check_lowered_program(self, smaller_physical_pitch: bool) -> None:
        torch.manual_seed(21)
        declared_seqs = self.MAX_SEQS + 2 if smaller_physical_pitch else self.MAX_SEQS
        segments = 2 * declared_seqs
        pitch = self.MAX_SEQS + 1
        manifest = self.manifest(
            max_seqs=declared_seqs,
            max_segments=segments,
        )
        metadata = offgraph_kv_constant_methods(manifest)
        self.assertEqual(metadata["get_offgraph_kv_max_seqs"], declared_seqs)
        lowered = self._lowered(manifest)
        runtime = {}
        for name, value in list(lowered.constants.items()):
            if name.startswith(OFFGRAPH_KV_FQN_PREFIX):
                runtime[name] = torch.zeros(
                    value.shape, device="cuda", dtype=value.dtype
                )
                if smaller_physical_pitch and name not in (
                    OFFGRAPH_KV_SEQ_COLS_FQN,
                    OFFGRAPH_KV_SEQ_INFO_FQN,
                ):
                    rows = value.shape[1] // (declared_seqs + 1)
                    runtime[name][:, rows * pitch :].fill_(float("nan"))
                lowered.constants[name] = runtime[name]
        module = lowered.module()
        oracle = _Oracle(4 * self.CAPACITY, (0, self.WINDOW))
        self.addCleanup(oracle.close)
        columns = {seq: self.MAX_SEQS - 1 - seq for seq in range(self.MAX_SEQS)}
        for groups in _STEPS:
            seq_ids, q, k, v, position = _batch(groups)
            flat, ring = oracle.step(seq_ids, q, k, v, position)
            seq_cols, seq_info = _plan(
                seq_ids,
                position,
                columns,
                self.MAX_WRITE,
                segments,
                pitch - 1,
            )
            runtime[OFFGRAPH_KV_SEQ_COLS_FQN].copy_(seq_cols)
            runtime[OFFGRAPH_KV_SEQ_INFO_FQN].copy_(seq_info)
            out = module(q, k, v, position.reshape(-1, 1))
            self.assertLess(_max_abs_diff(out, flat + ring), 3e-2)
            if smaller_physical_pitch:
                for name, value in runtime.items():
                    if name not in (OFFGRAPH_KV_SEQ_COLS_FQN, OFFGRAPH_KV_SEQ_INFO_FQN):
                        rows = value.shape[1] // (declared_seqs + 1)
                        self.assertTrue(torch.isnan(value[:, rows * pitch :]).all())

    def test_lowered_program_matches_the_neutral_op(self) -> None:
        self._check_lowered_program(False)

    def test_lowered_runtime_pitch_matches_the_neutral_op(self) -> None:
        self._check_lowered_program(True)

    def test_non_causal_context_window_reaches_sdpa_varlen(self) -> None:
        lowered = self._lowered(
            self.manifest(causal=False, context_window=True),
        )
        calls = [
            n
            for n in lowered.graph_module.graph.nodes
            if n.target == torch.ops.triton.sdpa_varlen.default
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertFalse(call.args[5])
            flag = call.kwargs.get(
                "context_window",
                call.args[9] if len(call.args) > 9 else None,
            )
            self.assertTrue(flag)

    def _lowered_draft_ring(self, block, window):
        writes = 3 + 2 * block
        q, k, v, position = _inputs(0, writes)
        q = q[:, :, : 2 * block].contiguous()
        program = torch.export.export(
            _DraftRing(), (q, k, v, position.reshape(-1, 1)), strict=False
        )
        edge = to_edge(
            program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        ).exported_program()
        manifest = self.manifest(
            causal=False,
            symmetric_window=True,
            max_write=64,
            maximum_capacity=128,
            layers=[{"layer_id": 0, "policy": "ring", "window": window}],
        )
        lowered = LowerOffGraphKVPass(parse_offgraph_kv_manifest(manifest))(edge)
        runtime = {}
        for name, value in list(lowered.constants.items()):
            if name.startswith(OFFGRAPH_KV_FQN_PREFIX):
                runtime[name] = torch.zeros_like(value, device="cuda")
                lowered.constants[name] = runtime[name]
        calls = [
            n
            for n in lowered.graph_module.graph.nodes
            if n.target == torch.ops.triton.sdpa_varlen.default
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertTrue(
            call.kwargs.get(
                "symmetric_window", call.args[10] if len(call.args) > 10 else False
            )
        )
        return lowered.module(), runtime, q

    def _draft_runtime_inputs(self, q, runtime, values, starts, block):
        rows = values.shape[0]
        slots = torch.arange(rows, device="cuda")
        positions, segments, columns = [], [], []
        for col, (start, fresh) in enumerate(zip(starts, (2, 1))):
            context = start + fresh
            retained = slots + rows * ((start - 1 - slots) // rows)
            values[:, col] = retained[:, None, None] / 16
            positions.extend(range(start, context + block))
            columns.extend([col] * (fresh + block))
            segments.append((col * block, block, col, context + block))
        position = torch.tensor(positions, device="cuda")
        k = torch.zeros(
            1, N_KV_HEADS, len(positions), HEAD_DIM, device="cuda", dtype=q.dtype
        )
        v = (position[None, None, :, None] / 16).expand_as(k).to(q.dtype).contiguous()
        runtime[OFFGRAPH_KV_SEQ_COLS_FQN][: len(columns)].copy_(torch.tensor(columns))
        runtime[OFFGRAPH_KV_SEQ_INFO_FQN].copy_(
            torch.tensor(
                [(*segment, self.MAX_SEQS + 1) for segment in segments]
                + [(0, 0, 0, 0, self.MAX_SEQS + 1)] * (self.MAX_SEGMENTS - 2),
                dtype=torch.int32,
            )
        )
        return k, v, position, segments

    def test_lowered_dflash_ring_matches_model_across_wrap(self):
        window = 3
        for block in (2, 8):
            module, runtime, q = self._lowered_draft_ring(block, window)
            rows = ring_physical_capacity(window, 64)
            values = runtime[f"{OFFGRAPH_KV_FQN_PREFIX}layer_0_v"].view(
                rows, self.MAX_SEQS + 1, N_KV_HEADS, HEAD_DIM
            )
            for starts in ((0, 0), (20, 13), (70, 90)):
                k, v, position, segments = self._draft_runtime_inputs(
                    q, runtime, values, starts, block
                )
                got = module(q, k, v, position.reshape(-1, 1))
                want = _draft_model_reference(q, values, segments, window, rows)
                torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)
                for col, (start, fresh) in enumerate(zip(starts, (2, 1))):
                    write_positions = torch.arange(
                        start, start + fresh + block, device="cuda"
                    )
                    torch.testing.assert_close(
                        values[write_positions % rows, col, 0, 0],
                        (write_positions / 16).to(q.dtype),
                    )

    def test_step_width_pass_accepts_the_packed_layout(self) -> None:
        with self.assertRaises(ValueError):
            CheckOffGraphKVStepWidthPass(b"0:0", layout="paged")
        CheckOffGraphKVStepWidthPass(b"0:0", layout="batched-sequence")


if __name__ == "__main__":
    unittest.main()
