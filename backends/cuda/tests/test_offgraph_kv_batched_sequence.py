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

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes.lower_offgraph_kv import (
    batched_sequence_max_segments,
    batched_sequence_step,
    CheckOffGraphKVStepWidthPass,
    LowerOffGraphKVPass,
    OFFGRAPH_KV_COMPILE_SPEC,
    OFFGRAPH_KV_FQN_PREFIX,
    OFFGRAPH_KV_SEQ_COLS_FQN,
    OFFGRAPH_KV_SEQ_INFO_FQN,
    OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC,
    parse_offgraph_kv_manifest,
    ring_physical_capacity,
)
from executorch.backends.cuda.tests.host_memory_test_utils import (
    IS_LINUX,
    PeakResidentBytes,
)

from executorch.examples.models.muse_glimmer.model.dflash_model import (
    build_dflash_swa_mask,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
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
                segments + [(0, 0, 0, 0)] * (self.MAX_SEGMENTS - 2), dtype=torch.int32
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


class _FlatLayer(torch.nn.Module):
    def forward(self, q, k, v, position):
        return torch.ops.kvcache.update_and_attend(
            q, k, v, position, 0, SCALE, torch.bfloat16
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(IS_LINUX, "reads /proc/self/statm")
class BatchedSequenceCompileHostMemoryTest(unittest.TestCase):
    """Lowering one batched-sequence layer grows host memory by well under
    the size of its declared KV storage, which the runtime supplies.
    """

    CAPACITY = 1 << 16
    MAX_SEQS = 63
    MAX_WRITE = 8

    def test_declared_storage_stays_off_the_host(self) -> None:
        # K and V of one layer: 1 GiB each. Dense zeros would grow the host by
        # the whole 2 GiB; the limit leaves the compile itself half of that.
        declared = 2 * self.CAPACITY * (self.MAX_SEQS + 1) * N_KV_HEADS * HEAD_DIM * 2
        limit = declared // 2
        # 8 GiB: fits the 24 GB A10G that runs unittest-cuda, and A100/H100.
        priming_bytes = 4 * declared
        free, _ = torch.cuda.mem_get_info()
        if free < priming_bytes:
            self.skipTest(
                f"needs {priming_bytes >> 30} GiB of free GPU memory to prime the "
                f"allocator's peak, has {free >> 30} GiB"
            )
        manifest = json.dumps(
            {
                "version": 1,
                "layout": "batched-sequence",
                "maximum_capacity": self.CAPACITY,
                "max_write": self.MAX_WRITE,
                "max_seqs": self.MAX_SEQS,
                "max_segments": batched_sequence_max_segments(self.MAX_SEQS),
                "layers": [{"layer_id": 0, "policy": "flat"}],
            }
        ).encode()
        q, k, v, position = _inputs(0, self.MAX_WRITE)
        t = torch.export.Dim("t", min=2, max=self.MAX_WRITE)
        program = torch.export.export(
            _FlatLayer(),
            (q, k, v, position.reshape(-1, 1)),
            dynamic_shapes=({2: t}, {2: t}, {2: t}, {0: t}),
            strict=False,
        )
        # Inductor's autotuner protects a kernel's written arguments from its
        # benchmark launches: it clones each on the GPU, or copies it to pinned
        # host memory when the clone would push GPU allocation past its
        # recorded peak. The storage is written, so in a fresh process the
        # copy would land in host memory and be measured here. Raising the
        # peak first keeps the clones on the GPU, as a model's weights would.
        # Freeing the block keeps the recorded peak but not the reservation,
        # and the cleanup resets the peak for later tests in this process.
        self.addCleanup(torch.cuda.reset_peak_memory_stats)
        priming = torch.empty(priming_bytes, dtype=torch.uint8, device="cuda")
        del priming
        torch.cuda.empty_cache()
        import torch._inductor.config as inductor_config

        with inductor_config.patch(
            {"aot_inductor.precompile_headers": False}
        ), PeakResidentBytes() as resident:
            lowered = to_edge_transform_and_lower(
                {"forward": program},
                partitioner={
                    "forward": [
                        CudaPartitioner(
                            [
                                CudaBackend.generate_method_name_compile_spec(
                                    "forward"
                                ),
                                CompileSpec(OFFGRAPH_KV_COMPILE_SPEC, manifest),
                                CompileSpec(
                                    OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC, b"3:0"
                                ),
                                CompileSpec("low_memory_mode", b"ON"),
                            ]
                        )
                    ]
                },
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
        graph = lowered.exported_program("forward").graph
        self.assertTrue(
            any("executorch_call_delegate" in str(n.target) for n in graph.nodes)
        )
        self.assertLess(
            resident.growth,
            limit,
            f"compiling grew the host by {resident.growth >> 20} MiB, limit "
            f"{limit >> 20} MiB, for {declared >> 20} MiB of declared KV storage",
        )


if __name__ == "__main__":
    unittest.main()
