# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
import unittest.mock

import torch

# The very functions the lowering pass traces and splices into the graph, so a
# passing test cannot agree with a decomposition the pass does not emit.
from executorch.backends.cuda.passes.lower_offgraph_kv import (
    LowerOffGraphKVPass,
    OFFGRAPH_KV_FQN_PREFIX,
    offgraph_step,
    parse_offgraph_kv_manifest,
    ring_attention_mask,
    ring_physical_capacity,
)
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.llm.cache.reference_cache import (
    CacheConfig,
    LayerPolicy,
    SequenceReferenceCache,
)

# Importing the op module registers kvcache::update_and_attend and exposes the
# registry the eager implementation reads its cache from.
from executorch.extension.llm.cache.update_and_attend import REGISTRY


N_KV_HEADS = 2
N_Q_HEADS = 4
HEAD_DIM = 64
SCALE = 0.125


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        raise unittest.SkipTest("BF16 not supported")


class _Oracle:
    """``kvcache::update_and_attend`` itself, stepped alongside the decomposition.

    The op is the contract every backend implements, so driving it here checks
    the decomposition against the same placement and masking rules the other
    backends are checked against -- rather than against a mask rebuilt in this
    file, which can only ever agree with itself.

    It is stateful in the same way the cache is: feed it each step's k/v and it
    keeps the history, so callers do not accumulate one.
    """

    def __init__(self, capacity: int, window: int = 0) -> None:
        self._cache = SequenceReferenceCache(
            CacheConfig(
                n_layers=1,
                n_kv_heads=N_KV_HEADS,
                head_dim=HEAD_DIM,
                capacity=capacity,
                dtype=torch.float32,
                layers=(LayerPolicy.ring(window) if window else LayerPolicy.flat(),),
            )
        )
        self._key = f"cuda-offgraph-decomp-{id(self)}"
        REGISTRY.install(self._key, self._cache)

    def step(self, q, k, v, position) -> torch.Tensor:
        # The reference keeps its history on CPU in float32; position is
        # [q_len, n_dims] per the op contract.
        with REGISTRY.active(self._key):
            return torch.ops.kvcache.update_and_attend(
                q.float().cpu(),
                k.float().cpu(),
                v.float().cpu(),
                position.reshape(-1, 1).cpu(),
                0,
                SCALE,
                torch.float32,
            )

    def close(self) -> None:
        REGISTRY.uninstall(self._key)


def _storage(capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    # BSHD, as LowerOffGraphKVPass declares it.
    shape = (1, capacity, N_KV_HEADS, HEAD_DIM)
    return (
        torch.zeros(shape, device="cuda", dtype=torch.bfloat16),
        torch.zeros(shape, device="cuda", dtype=torch.bfloat16),
    )


def _inputs(start: int, length: int):
    q = torch.randn(1, N_Q_HEADS, length, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        1, N_KV_HEADS, length, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    position = torch.arange(start, start + length, device="cuda")
    return q, k, v, position


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float().cpu() - expected).abs().max().item()


class OffGraphKVDecompositionTest(unittest.TestCase):
    """index_copy_ + triton::sdpa must match the neutral op it replaced."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_cuda()

    def _assert_matches(self, out, oracle, q, k, v, position) -> None:
        self.assertLess(_max_abs_diff(out, oracle.step(q, k, v, position)), 1e-2)

    def test_flat_prefill_then_decode_matches_oracle(self) -> None:
        torch.manual_seed(0)
        capacity = 128
        k_storage, v_storage = _storage(capacity)
        oracle = _Oracle(capacity)
        self.addCleanup(oracle.close)

        # A prefill chunk, then single-token decodes: kv_len has to track the
        # sequence while the buffer's declared shape stays at capacity.
        for start, length in ((0, 24), (24, 1), (25, 1), (26, 8)):
            q, k, v, position = _inputs(start, length)
            out = offgraph_step(
                q, k, v, position, k_storage, v_storage, SCALE, capacity
            )
            self._assert_matches(out, oracle, q, k, v, position)

    def test_flat_write_lands_at_logical_position(self) -> None:
        torch.manual_seed(1)
        k_storage, v_storage = _storage(64)
        q, k, v, position = _inputs(8, 8)

        offgraph_step(q, k, v, position, k_storage, v_storage, SCALE, 64)

        self.assertTrue(torch.equal(k_storage[:, 8:16], k.transpose(1, 2)))
        self.assertTrue(torch.equal(v_storage[:, 8:16], v.transpose(1, 2)))
        # Untouched slots stay untouched, so a stride mistake that scattered
        # the write would show up here.
        self.assertFalse(k_storage[:, 16:].any())
        self.assertFalse(k_storage[:, :8].any())

    def test_flat_accepts_the_neutral_op_position_shape(self) -> None:
        # kvcache::update_and_attend documents position as [q_len, n_dims].
        torch.manual_seed(6)
        k_storage, v_storage = _storage(64)
        oracle = _Oracle(64)
        self.addCleanup(oracle.close)
        q, k, v, position = _inputs(0, 5)

        out = offgraph_step(
            q, k, v, position.reshape(-1, 1), k_storage, v_storage, SCALE, 64
        )

        self._assert_matches(out, oracle, q, k, v, position)

    def test_bound_carries_the_step_width_from_the_shape(self) -> None:
        # AOTI autotunes sdpa on generated inputs whose integer data is zero.
        # The bound must still reflect a full step then, or sdpa is tuned for a
        # one-token context and long prefill runs on a slower tile.
        k_storage, v_storage = _storage(64)
        q, k, v, _ = _inputs(0, 16)
        seen = []
        real_sdpa = torch.ops.triton.sdpa

        def spy(*args):
            seen.append(int(args[-1]))
            return real_sdpa(*args)

        with unittest.mock.patch.object(torch.ops.triton, "sdpa", spy):
            offgraph_step(
                q,
                k,
                v,
                torch.zeros(16, dtype=torch.long, device=q.device),
                k_storage,
                v_storage,
                SCALE,
                64,
            )

        self.assertEqual(seen, [16])

    def test_flat_runs_over_an_allocation_smaller_than_declared(self) -> None:
        # The runtime binds the full declared shape over a buffer that only
        # holds the rows grown so far. Every access must stay inside that
        # buffer: poison the region past it and require it untouched, and the
        # attention to still match.
        torch.manual_seed(7)
        declared, allocated = 4096, 512
        row = N_KV_HEADS * HEAD_DIM
        k_buf = torch.full(
            (declared * row,), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        v_buf = torch.full_like(k_buf, float("nan"))
        k_buf[: allocated * row].zero_()
        v_buf[: allocated * row].zero_()
        k_storage = k_buf.view(1, declared, N_KV_HEADS, HEAD_DIM)
        v_storage = v_buf.view(1, declared, N_KV_HEADS, HEAD_DIM)
        oracle = _Oracle(declared)
        self.addCleanup(oracle.close)

        # Crosses sdpa's split-K threshold, then decodes.
        for start, length in ((0, 300), (300, 1), (301, 1), (302, 64)):
            q, k, v, position = _inputs(start, length)
            out = offgraph_step(
                q, k, v, position, k_storage, v_storage, SCALE, declared
            )
            self._assert_matches(out, oracle, q, k, v, position)

        self.assertTrue(torch.isnan(k_buf[allocated * row :]).all())
        self.assertTrue(torch.isnan(v_buf[allocated * row :]).all())

    def test_flat_decode_over_large_buffer_matches_oracle(self) -> None:
        # Past _SPLITK_LKV_THRESHOLD a decode with kv_len routes through the
        # split-K path inside sdpa, which is a different kernel.
        torch.manual_seed(2)
        capacity = 512
        k_storage, v_storage = _storage(capacity)
        oracle = _Oracle(capacity)
        self.addCleanup(oracle.close)

        q, k, v, position = _inputs(0, 257)
        offgraph_step(q, k, v, position, k_storage, v_storage, SCALE, capacity)
        oracle.step(q, k, v, position)

        q, k, v, position = _inputs(257, 1)
        out = offgraph_step(q, k, v, position, k_storage, v_storage, SCALE, capacity)
        self._assert_matches(out, oracle, q, k, v, position)

    def _run_ring(self, window: int, max_write: int, steps) -> None:
        buf_size = ring_physical_capacity(window, max_write)
        k_storage, v_storage = _storage(buf_size)
        # The oracle bounds the logical sequence, not the ring's slots.
        oracle = _Oracle(capacity=4096, window=window)
        self.addCleanup(oracle.close)

        for start, length in steps:
            q, k, v, position = _inputs(start, length)
            mask = ring_attention_mask(position, buf_size, window)
            out = offgraph_step(
                q, k, v, position, k_storage, v_storage, SCALE, buf_size, mask
            )
            self._assert_matches(out, oracle, q, k, v, position)

    def test_ring_wrap_preserves_logical_attention_order(self) -> None:
        torch.manual_seed(3)
        self._run_ring(window=16, max_write=16, steps=((0, 12), (12, 12), (24, 12)))

    def test_ring_decode_after_wrap_matches_oracle(self) -> None:
        # Single-token steps past the wrap: every slot holds a position from a
        # different lap, which is exactly what ring_pos has to recover.
        torch.manual_seed(4)
        steps = [(0, 20)] + [(20 + i, 1) for i in range(12)]
        self._run_ring(window=16, max_write=16, steps=tuple(steps))

    def test_ring_max_write_step_straddles_a_wrap(self) -> None:
        # A step of max_write tokens starting past the window is the chunked
        # prefill case: its earliest query reads back window-1 positions before
        # the step, so those must survive the step's own writes, and queries
        # within the one step land on opposite sides of the wrap.
        torch.manual_seed(5)
        window = 16
        max_write = 2 * window
        self._run_ring(
            window=window,
            max_write=max_write,
            steps=((0, max_write), (max_write, max_write), (2 * max_write, max_write)),
        )

    def test_ring_mask_marks_only_live_in_window_slots(self) -> None:
        # Guards the mask directly, so a mask bug cannot hide behind attention
        # numerics that happen to agree within tolerance.
        window, max_write = 4, 4
        buf_size = ring_physical_capacity(window, max_write)  # 7
        position = torch.arange(8, 10, device="cuda")  # total_written = 10
        mask = ring_attention_mask(position, buf_size, window)

        self.assertEqual(tuple(mask.shape), (1, 1, 2, buf_size))
        # Slot j holds the newest logical position congruent to j mod 7.
        held = [7, 8, 9, 3, 4, 5, 6]
        for row, query_pos in enumerate((8, 9)):
            expected = [0 <= query_pos - p < window for p in held]  # causal + sliding
            self.assertEqual(mask[0, 0, row].tolist(), expected)


class _FlatAndRing(torch.nn.Module):
    def forward(self, q, k, v, position):
        flat = torch.ops.kvcache.update_and_attend(
            q, k, v, position, 0, SCALE, torch.bfloat16
        )
        ring = torch.ops.kvcache.update_and_attend(
            q, k, v, position, 1, SCALE, torch.bfloat16
        )
        return flat + ring


class LowerOffGraphKVPassTest(unittest.TestCase):
    """The pass itself, on an exported program, rather than its building blocks."""

    CAPACITY = 64
    MAX_WRITE = 8
    WINDOW = 4

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_cuda()

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
        manifest = parse_offgraph_kv_manifest(
            json.dumps(
                {
                    "version": 1,
                    "maximum_capacity": self.CAPACITY,
                    "max_write": self.MAX_WRITE,
                    "layers": [
                        {"layer_id": 0, "policy": "flat"},
                        {"layer_id": 1, "policy": "ring", "window": self.WINDOW},
                    ],
                }
            ).encode()
        )
        return LowerOffGraphKVPass(manifest)(edge)

    def test_replaces_the_neutral_op_with_sequence_major_storage(self) -> None:
        lowered = self._lowered()
        graph = lowered.graph_module.graph
        target = exir_ops.edge.kvcache.update_and_attend.default

        self.assertFalse(any(n.target == target for n in graph.nodes))
        storage = {
            n.name: tuple(n.meta["val"].shape)
            for n in graph.nodes
            if n.op == "placeholder" and n.name.startswith(OFFGRAPH_KV_FQN_PREFIX)
        }
        ring = ring_physical_capacity(self.WINDOW, self.MAX_WRITE)
        flat_shape = (1, self.CAPACITY, N_KV_HEADS, HEAD_DIM)
        ring_shape = (1, ring, N_KV_HEADS, HEAD_DIM)
        # No capacity constant: nothing in the decomposed graph reads one.
        self.assertEqual(
            storage,
            {
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_0_k": flat_shape,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_0_v": flat_shape,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_1_k": ring_shape,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_1_v": ring_shape,
            },
        )
        # Payloads stay out of the artifact; the runtime supplies storage.
        for name in storage:
            self.assertEqual(lowered.constants[name].untyped_storage().nbytes(), 0)
        # One step serves both layers: the flat one attends with sdpa's
        # device-side causal and no mask tensor, the ring one behind its mask.
        sdpa_calls = [
            n for n in graph.nodes if n.target == torch.ops.triton.sdpa.default
        ]
        self.assertEqual(
            sorted((n.args[3] is None, n.args[5]) for n in sdpa_calls),
            [(False, False), (True, True)],
        )

    def test_lowered_program_matches_the_neutral_op(self) -> None:
        torch.manual_seed(8)
        lowered = self._lowered()
        for name, value in list(lowered.constants.items()):
            if name.startswith(OFFGRAPH_KV_FQN_PREFIX):
                lowered.constants[name] = torch.zeros(
                    value.shape, device="cuda", dtype=value.dtype
                )
        module = lowered.module()
        flat = _Oracle(self.CAPACITY)
        ring = _Oracle(self.CAPACITY, window=self.WINDOW)
        self.addCleanup(flat.close)
        self.addCleanup(ring.close)

        # Prefill, decodes, then a chunk that wraps the ring.
        for start, length in ((0, 8), (8, 1), (9, 1), (10, 6)):
            q, k, v, position = _inputs(start, length)
            out = module(q, k, v, position.reshape(-1, 1))
            expected = flat.step(q, k, v, position) + ring.step(q, k, v, position)
            self.assertLess(_max_abs_diff(out, expected), 2e-2)
