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

from executorch.backends.cuda.passes.lower_offgraph_kv import (
    batched_sequence_step,
    CheckOffGraphKVStepWidthPass,
    LowerOffGraphKVPass,
    OFFGRAPH_KV_FQN_PREFIX,
    OFFGRAPH_KV_SEQ_COLS_FQN,
    OFFGRAPH_KV_SEQ_INFO_FQN,
    parse_offgraph_kv_manifest,
    ring_physical_capacity,
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
    and one (q_start, q_len, col, kv_end) row per run of one sequence."""
    seq_cols = [columns[s] for s in seq_ids] + [scratch] * (width - len(seq_ids))
    info, start = [], 0
    positions = position.tolist()
    while start < len(seq_ids):
        end = start
        while end < len(seq_ids) and seq_ids[end] == seq_ids[start]:
            end += 1
        info.append(
            (start, end - start, columns[seq_ids[start]], positions[end - 1] + 1)
        )
        start = end
    info += [(0, 0, 0, 0)] * (max_segments - len(info))
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
    return (actual.float().cpu() - expected).abs().max().item()


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

    def _run(self, steps, capacity, window=0, max_segments=6):
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
        k_store = torch.zeros(
            1, rows * cols, N_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        v_store = torch.zeros_like(k_store)
        for groups in steps:
            seq_ids, q, k, v, position = _batch(groups)
            (expected,) = oracle.step(seq_ids, q, k, v, position)
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
                rows,
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

    def _lowered(self):
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
        return LowerOffGraphKVPass(parse_offgraph_kv_manifest(self.manifest()))(edge)

    def test_manifest_is_validated(self) -> None:
        parsed = parse_offgraph_kv_manifest(self.manifest())
        self.assertEqual(parsed["layout"], "batched-sequence")
        self.assertTrue(parsed["causal"])
        self.assertFalse(
            parse_offgraph_kv_manifest(self.manifest(causal=False))["causal"]
        )
        for bad in (
            {"max_seqs": 0},
            {"max_segments": None},
            {"causal": 1},
            {"max_write": self.CAPACITY + 1},
        ):
            with self.assertRaises(ValueError):
                parse_offgraph_kv_manifest(self.manifest(**bad))

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
                OFFGRAPH_KV_SEQ_INFO_FQN: (self.MAX_SEGMENTS, 4),
            },
        )
        calls = [
            n for n in graph.nodes if n.target == torch.ops.triton.sdpa_varlen.default
        ]
        self.assertEqual(len(calls), 2)

    def test_lowered_program_matches_the_neutral_op(self) -> None:
        torch.manual_seed(21)
        lowered = self._lowered()
        runtime = {}
        for name, value in list(lowered.constants.items()):
            if name.startswith(OFFGRAPH_KV_FQN_PREFIX):
                runtime[name] = torch.zeros(
                    value.shape, device="cuda", dtype=value.dtype
                )
                lowered.constants[name] = runtime[name]
        module = lowered.module()
        oracle = _Oracle(4 * self.CAPACITY, (0, self.WINDOW))
        self.addCleanup(oracle.close)
        columns = {seq: seq for seq in range(self.MAX_SEQS)}
        for groups in _STEPS:
            seq_ids, q, k, v, position = _batch(groups)
            flat, ring = oracle.step(seq_ids, q, k, v, position)
            seq_cols, seq_info = _plan(
                seq_ids,
                position,
                columns,
                self.MAX_WRITE,
                self.MAX_SEGMENTS,
                self.MAX_SEQS,
            )
            runtime[OFFGRAPH_KV_SEQ_COLS_FQN].copy_(seq_cols)
            runtime[OFFGRAPH_KV_SEQ_INFO_FQN].copy_(seq_info)
            out = module(q, k, v, position.reshape(-1, 1))
            self.assertLess(_max_abs_diff(out, flat + ring), 3e-2)

    def test_step_width_pass_accepts_the_packed_layout(self) -> None:
        with self.assertRaises(ValueError):
            CheckOffGraphKVStepWidthPass(b"0:0", layout="paged")
        CheckOffGraphKVStepWidthPass(b"0:0", layout="batched-sequence")


if __name__ == "__main__":
    unittest.main()
