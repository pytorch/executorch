# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import unittest
import unittest.mock

import torch

# The very functions the lowering pass traces and splices into the graph, so a
# passing test cannot agree with a decomposition the pass does not emit.
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.passes.lower_offgraph_kv import (
    cell_step,
    CheckOffGraphKVStepWidthPass,
    LowerOffGraphKVPass,
    OFFGRAPH_KV_CELLS_FQN,
    OFFGRAPH_KV_COMPILE_SPEC,
    OFFGRAPH_KV_FQN_PREFIX,
    OFFGRAPH_KV_READ_LEN_FQN,
    OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC,
    offgraph_kv_mask_fqn,
    offgraph_step,
    parse_offgraph_kv_manifest,
    ring_attention_mask,
    ring_physical_capacity,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.extension.llm.cache.reference_cache import (
    CacheConfig,
    CellReferenceCache,
    LayerPolicy,
    SequenceReferenceCache,
)

# Importing the op module registers kvcache::update_and_attend and exposes the
# registry the eager implementation reads its cache from.
from executorch.extension.llm.cache.update_and_attend import REGISTRY

# The module, not the op the package re-exports under the same name: the tests
# spy on which kernel its dispatch launches.
_SDPA = importlib.import_module("executorch.backends.cuda.triton.kernels.sdpa")


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
    def __init__(self, out_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.out_dtype = out_dtype

    def forward(self, q, k, v, position):
        flat = torch.ops.kvcache.update_and_attend(
            q, k, v, position, 0, SCALE, self.out_dtype
        )
        ring = torch.ops.kvcache.update_and_attend(
            q, k, v, position, 1, SCALE, self.out_dtype
        )
        return flat + ring


class _Flat(torch.nn.Module):
    def forward(self, q, k, v, position):
        return torch.ops.kvcache.update_and_attend(
            q, k, v, position, 0, SCALE, torch.bfloat16
        )


class CheckOffGraphKVStepWidthPassTest(unittest.TestCase):
    """The runtime reads the step width where the spec says; it must be there."""

    MAX_WRITE = 8

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_cuda()

    def _edge(self):
        q, k, v, position = _inputs(0, self.MAX_WRITE)
        t = torch.export.Dim("t", min=1, max=self.MAX_WRITE)
        program = torch.export.export(
            _FlatAndRing(),
            (q, k, v, position.reshape(-1, 1)),
            dynamic_shapes=({2: t}, {2: t}, {2: t}, {0: t}),
            strict=False,
        )
        return to_edge(
            program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        ).exported_program()

    def test_accepts_the_position_input_and_its_length_dim(self) -> None:
        edge = self._edge()
        self.assertIs(CheckOffGraphKVStepWidthPass(b"3:0")(edge), edge)

    def test_rejects_an_input_that_does_not_feed_position(self) -> None:
        # Input 1 is k: its dim 2 is the step length, but it is not position.
        with self.assertRaisesRegex(ValueError, "position comes from"):
            CheckOffGraphKVStepWidthPass(b"1:2")(self._edge())

    def test_rejects_a_dim_that_is_not_the_step_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "not the step"):
            CheckOffGraphKVStepWidthPass(b"3:1")(self._edge())

    def test_rejects_an_input_past_the_delegate_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "the delegate takes 4"):
            CheckOffGraphKVStepWidthPass(b"4:0")(self._edge())

    def test_rejects_a_malformed_spec(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_index:dim"):
            CheckOffGraphKVStepWidthPass(b"1")


class LowerOffGraphKVPassTest(unittest.TestCase):
    """The pass itself, on an exported program, rather than its building blocks."""

    CAPACITY = 64
    MAX_WRITE = 8
    WINDOW = 4

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_cuda()

    def _lowered(self, module=None, args=None):
        if args is None:
            q, k, v, position = _inputs(0, self.MAX_WRITE)
            args = (q, k, v, position.reshape(-1, 1))
        t = torch.export.Dim("t", min=1, max=self.MAX_WRITE)
        program = torch.export.export(
            module if module is not None else _FlatAndRing(),
            args,
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

    def _run_once(self, lowered, args):
        for name, value in list(lowered.constants.items()):
            if name.startswith(OFFGRAPH_KV_FQN_PREFIX):
                lowered.constants[name] = torch.zeros(
                    value.shape, device="cuda", dtype=value.dtype
                )
        return lowered.module()(*args)

    def test_lowered_output_follows_out_dtype(self) -> None:
        # sdpa returns the query's dtype; the op promises out_dtype.
        q, k, v, position = _inputs(0, self.MAX_WRITE)
        args = (q, k, v, position.reshape(-1, 1))
        wide = self._run_once(self._lowered(_FlatAndRing(torch.float32), args), args)
        narrow = self._run_once(self._lowered(_FlatAndRing(), args), args)

        self.assertEqual(wide.dtype, torch.float32)
        self.assertEqual(narrow.dtype, torch.bfloat16)
        self.assertLess((wide - narrow.float()).abs().max().item(), 2e-2)

    def test_rejects_value_head_dim_unlike_key_head_dim(self) -> None:
        q, k, _, position = _inputs(0, self.MAX_WRITE)
        v = torch.randn(*k.shape[:-1], HEAD_DIM * 2, device="cuda", dtype=k.dtype)

        with self.assertRaisesRegex(ValueError, "v_head_dim == head_dim"):
            self._lowered(_Flat(), (q, k, v, position.reshape(-1, 1)))

    def test_rejects_multi_component_positions(self) -> None:
        q, k, v, position = _inputs(0, self.MAX_WRITE)
        two_d = torch.stack((position, position), dim=1)

        with self.assertRaisesRegex(ValueError, "one position component"):
            self._lowered(_Flat(), (q, k, v, two_d))

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


# -- cell layout ---------------------------------------------------------------


class _CellOracle:
    """The neutral op over ``CellReferenceCache``: the placement and masking
    contract the cell layout implements, for several sequences at once.

    One layer per window, so a test can drive flat and windowed layers of the
    same step. After a step it also exposes that step's plan -- the cells it
    placed and the per-window masks -- which is exactly what the runtime cache
    writes into the lowered program's step buffers.
    """

    def __init__(self, max_cells: int, windows=(0,)) -> None:
        self.windows = tuple(windows)
        self._cache = CellReferenceCache(
            CacheConfig(
                n_layers=len(self.windows),
                n_kv_heads=N_KV_HEADS,
                head_dim=HEAD_DIM,
                capacity=max_cells,
                dtype=torch.float32,
                layers=tuple(
                    LayerPolicy.ring(w) if w else LayerPolicy.flat()
                    for w in self.windows
                ),
            )
        )
        self._key = f"cuda-offgraph-cell-{id(self)}"
        REGISTRY.install(self._key, self._cache)

    def step(self, seq_ids, q, k, v, position):
        """Declares the step, runs every layer; returns one output per layer."""
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

    def plan(self):
        # The reference keeps the step's plan private; it is the same lowest-
        # free placement and ownership mask the C++ CellCache computes.
        plan = self._cache._plan
        return plan.cells, plan.base.shape[-1], plan.mask_for

    def close(self) -> None:
        REGISTRY.uninstall(self._key)


class _CellBuffers:
    """The runtime side of the cell layout: pools plus the step buffers."""

    def __init__(self, max_write: int, max_cells: int, windows=(0,)) -> None:
        self.windows = tuple(windows)
        self.pools = [_storage(max_cells) for _ in self.windows]
        self.cells = torch.zeros(max_write, dtype=torch.long, device="cuda")
        self.read_len = torch.zeros(1, dtype=torch.long, device="cuda")
        self.masks = {
            w: torch.zeros(1, 1, max_write, max_cells, dtype=torch.bool, device="cuda")
            for w in set(self.windows)
        }

    def load(self, plan) -> None:
        cells, read_len, mask_for = plan
        width = cells.numel()
        self.cells[:width] = cells.to("cuda")
        self.read_len.fill_(read_len)
        for window, mask in self.masks.items():
            mask.zero_()
            mask[0, 0, :width, :read_len] = mask_for(window).to("cuda")

    def step(self, layer: int, q, k, v):
        k_pool, v_pool = self.pools[layer]
        return cell_step(
            q,
            k,
            v,
            k_pool,
            v_pool,
            self.cells,
            self.read_len,
            self.masks[self.windows[layer]],
            SCALE,
        )


def _batch(groups):
    """Packs (seq_id, start, length) groups onto one token axis."""
    parts = [_inputs(start, length) for _, start, length in groups]
    seq_ids = [seq for seq, _, length in groups for _ in range(length)]
    q = torch.cat([p[0] for p in parts], dim=2)
    k = torch.cat([p[1] for p in parts], dim=2)
    v = torch.cat([p[2] for p in parts], dim=2)
    position = torch.cat([p[3] for p in parts])
    return seq_ids, q, k, v, position


# Two sequences prefilling in one step, decoding together, then one decoding
# beside the other's chunk: every combination a packed batch produces.
_INTERLEAVED = (
    ((0, 0, 5), (1, 0, 3)),
    ((0, 5, 1), (1, 3, 1)),
    ((1, 4, 4), (0, 6, 1)),
    ((0, 7, 1),),
)


class OffGraphKVCellDecompositionTest(unittest.TestCase):
    """cell_step, fed the reference cache's plan, must match the neutral op."""

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_cuda()

    def _run(self, steps, max_cells: int, windows=(0,), max_write: int = 16):
        oracle = _CellOracle(max_cells, windows)
        self.addCleanup(oracle.close)
        buffers = _CellBuffers(max_write, max_cells, windows)
        for groups in steps:
            seq_ids, q, k, v, position = _batch(groups)
            expected = oracle.step(seq_ids, q, k, v, position)
            buffers.load(oracle.plan())
            for layer in range(len(windows)):
                out = buffers.step(layer, q, k, v)
                self.assertLess(_max_abs_diff(out, expected[layer]), 1e-2)
        return buffers, oracle

    def test_interleaved_sequences_match_oracle(self) -> None:
        torch.manual_seed(10)
        self._run(_INTERLEAVED, max_cells=64)

    def test_windowed_layer_masks_older_cells(self) -> None:
        torch.manual_seed(11)
        steps = (((0, 0, 6), (1, 0, 6)),) + tuple(
            ((0, 6 + i, 1), (1, 6 + i, 1)) for i in range(5)
        )
        self._run(steps, max_cells=64, windows=(0, 4))

    def test_writes_land_in_placed_cells(self) -> None:
        torch.manual_seed(12)
        oracle = _CellOracle(64)
        self.addCleanup(oracle.close)
        buffers = _CellBuffers(16, 64)
        seq_ids, q, k, v, position = _batch(((0, 0, 3), (1, 0, 2)))
        oracle.step(seq_ids, q, k, v, position)
        plan = oracle.plan()
        buffers.load(plan)
        buffers.step(0, q, k, v)

        cells = plan[0].to("cuda")
        k_pool, v_pool = buffers.pools[0]
        self.assertTrue(torch.equal(k_pool[0, cells], k[0].transpose(0, 1)))
        self.assertTrue(torch.equal(v_pool[0, cells], v[0].transpose(0, 1)))
        untouched = torch.ones(64, dtype=torch.bool, device="cuda")
        untouched[cells] = False
        self.assertFalse(k_pool[0, untouched].any())

    def test_batched_sequence_sees_only_its_own_cells(self) -> None:
        # Independent of the oracle: a sequence's output is the same whether
        # or not another sequence shares the forward.
        torch.manual_seed(13)
        a = _inputs(0, 6)
        b = _inputs(0, 4)
        solo_oracle = _CellOracle(64)
        self.addCleanup(solo_oracle.close)
        solo = _CellBuffers(16, 64)
        solo_oracle.step([0] * 6, *a)
        solo.load(solo_oracle.plan())
        alone = solo.step(0, a[0], a[1], a[2])

        pair_oracle = _CellOracle(64)
        self.addCleanup(pair_oracle.close)
        pair = _CellBuffers(16, 64)
        q, k, v = (torch.cat([b[i], a[i]], dim=2) for i in range(3))
        position = torch.cat([b[3], a[3]])
        pair_oracle.step([1] * 4 + [0] * 6, q, k, v, position)
        pair.load(pair_oracle.plan())
        together = pair.step(0, q, k, v)

        self.assertLess(
            (together[:, :, 4:].float() - alone.float()).abs().max().item(), 1e-2
        )

    def _assert_read_len_bounds(self, steps, max_cells: int, max_write: int) -> None:
        # The runtime writes only the mask's [:width, :read_len] each step and
        # leaves stale columns past read_len from earlier, wider steps. Before
        # every step, poison everything past read_len -- mask true, pool NaN --
        # and require the output unchanged.
        oracle = _CellOracle(max_cells)
        self.addCleanup(oracle.close)
        buffers = _CellBuffers(max_write, max_cells)
        k_pool, v_pool = buffers.pools[0]
        for groups in steps:
            seq_ids, q, k, v, position = _batch(groups)
            (expected,) = oracle.step(seq_ids, q, k, v, position)
            plan = oracle.plan()
            buffers.load(plan)
            read_len = plan[1]
            buffers.masks[0][..., read_len:] = True
            k_pool[:, read_len:] = float("nan")
            v_pool[:, read_len:] = float("nan")
            out = buffers.step(0, q, k, v)
            self.assertFalse(torch.isnan(out).any(), groups)
            self.assertLess(_max_abs_diff(out, expected), 1e-2, groups)

    def test_read_len_bounds_the_sweep(self) -> None:
        torch.manual_seed(20)
        self._assert_read_len_bounds(
            (((0, 0, 5), (1, 0, 3)),), max_cells=64, max_write=16
        )

    def test_read_len_bounds_both_split_k_kernels(self) -> None:
        # sdpa takes split-K only for one, or two to four, query rows over a
        # pool past its threshold. After a long prefill, decode one token,
        # then three, so each split-K kernel runs over a poisoned tail.
        torch.manual_seed(21)
        steps = (
            ((0, 0, 290), (1, 0, 3)),
            ((0, 290, 1),),
            ((0, 291, 1), (1, 3, 2)),
        )
        with unittest.mock.patch.object(
            _SDPA, "_launch_decode_splitk", wraps=_SDPA._launch_decode_splitk
        ) as decode, unittest.mock.patch.object(
            _SDPA,
            "_launch_small_query_splitk",
            wraps=_SDPA._launch_small_query_splitk,
        ) as small_query:
            self._assert_read_len_bounds(steps, max_cells=512, max_write=296)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(small_query.call_count, 1)

    def test_split_k_decodes_over_a_large_pool(self) -> None:
        # read_len past sdpa's split-K threshold, with one, then three, query
        # rows: the decode and small-query split-K kernels under an explicit
        # mask, which no single-sequence path exercises.
        torch.manual_seed(14)
        steps = (
            ((0, 0, 200), (1, 0, 80)),
            ((2, 0, 16),),
            ((0, 200, 1),),
            ((0, 201, 1), (1, 80, 1), (2, 16, 1)),
        )
        self._run(steps, max_cells=512, max_write=296)


class LowerOffGraphKVCellPassTest(unittest.TestCase):
    """The pass in cell layout, on an exported program."""

    MAX_CELLS = 64
    MAX_WRITE = 8
    WINDOW = 4

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_cuda()

    @classmethod
    def manifest(cls) -> bytes:
        return json.dumps(
            {
                "version": 1,
                "layout": "cell",
                "maximum_capacity": cls.MAX_CELLS,
                "max_cells": cls.MAX_CELLS,
                "max_write": cls.MAX_WRITE,
                "layers": [
                    {"layer_id": 0, "policy": "flat"},
                    {"layer_id": 1, "policy": "ring", "window": cls.WINDOW},
                ],
            }
        ).encode()

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

    def test_manifest_layout_is_validated(self) -> None:
        base = json.loads(self.manifest())
        sequence = dict(base, layout="sequence")
        del sequence["max_cells"]
        self.assertEqual(
            parse_offgraph_kv_manifest(json.dumps(sequence).encode())["layout"],
            "sequence",
        )
        del sequence["layout"]
        self.assertEqual(
            parse_offgraph_kv_manifest(json.dumps(sequence).encode())["layout"],
            "sequence",
        )
        for bad in (
            dict(base, layout="paged"),
            {k: v for k, v in base.items() if k != "max_cells"},
            dict(base, max_cells=self.MAX_WRITE - 1),
            # A sequence step writes into one sequence's capacity.
            dict(sequence, max_write=sequence["maximum_capacity"] + 1),
        ):
            with self.assertRaises(ValueError):
                parse_offgraph_kv_manifest(json.dumps(bad).encode())
        # A cell step packs several sequences into the pool, so only the pool
        # bounds its width: two six-token sequences of capacity 8 fit 16 cells.
        packed = dict(base, maximum_capacity=8, max_cells=16, max_write=12)
        self.assertEqual(
            parse_offgraph_kv_manifest(json.dumps(packed).encode())["max_write"], 12
        )

    def test_declares_one_pool_per_layer_and_shared_step_buffers(self) -> None:
        lowered = self._lowered()
        graph = lowered.graph_module.graph
        target = exir_ops.edge.kvcache.update_and_attend.default
        self.assertFalse(any(n.target == target for n in graph.nodes))

        buffers = {
            n.name: tuple(n.meta["val"].shape)
            for n in graph.nodes
            if n.op == "placeholder" and n.name.startswith(OFFGRAPH_KV_FQN_PREFIX)
        }
        pool = (1, self.MAX_CELLS, N_KV_HEADS, HEAD_DIM)
        mask = (1, 1, self.MAX_WRITE, self.MAX_CELLS)
        # Both layers keep their history in a full pool; the window only picks
        # which mask the layer reads.
        self.assertEqual(
            buffers,
            {
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_0_k": pool,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_0_v": pool,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_1_k": pool,
                f"{OFFGRAPH_KV_FQN_PREFIX}layer_1_v": pool,
                OFFGRAPH_KV_CELLS_FQN: (self.MAX_WRITE,),
                OFFGRAPH_KV_READ_LEN_FQN: (1,),
                offgraph_kv_mask_fqn(0): mask,
                offgraph_kv_mask_fqn(self.WINDOW): mask,
            },
        )
        # The pools are declared without bytes; the step buffers are small and
        # carry zeros for the compiler. Both are runtime-owned.
        for name in buffers:
            nbytes = lowered.constants[name].untyped_storage().nbytes()
            if "layer_" in name:
                self.assertEqual(nbytes, 0, name)
            else:
                self.assertFalse(lowered.constants[name].any(), name)
        sdpa_calls = [
            n for n in graph.nodes if n.target == torch.ops.triton.sdpa.default
        ]
        # Always an explicit mask, never sdpa's own causal alignment.
        self.assertEqual(
            [(n.args[3] is not None, n.args[5]) for n in sdpa_calls],
            [(True, False), (True, False)],
        )

    def test_lowered_program_matches_the_neutral_op(self) -> None:
        torch.manual_seed(15)
        lowered = self._lowered()
        runtime = {}
        for name, value in list(lowered.constants.items()):
            if name.startswith(OFFGRAPH_KV_FQN_PREFIX):
                runtime[name] = torch.zeros(
                    value.shape, device="cuda", dtype=value.dtype
                )
                lowered.constants[name] = runtime[name]
        module = lowered.module()
        oracle = _CellOracle(self.MAX_CELLS, windows=(0, self.WINDOW))
        self.addCleanup(oracle.close)

        for groups in _INTERLEAVED:
            seq_ids, q, k, v, position = _batch(groups)
            flat, ring = oracle.step(seq_ids, q, k, v, position)
            cells, read_len, mask_for = oracle.plan()
            width = cells.numel()
            # What the runtime cache writes before the forward.
            runtime[OFFGRAPH_KV_CELLS_FQN][:width] = cells.to("cuda")
            runtime[OFFGRAPH_KV_READ_LEN_FQN].fill_(read_len)
            for window in (0, self.WINDOW):
                mask = runtime[offgraph_kv_mask_fqn(window)]
                mask.zero_()
                mask[0, 0, :width, :read_len] = mask_for(window).to("cuda")
            out = module(q, k, v, position.reshape(-1, 1))
            self.assertLess(_max_abs_diff(out, flat + ring), 2e-2)


class _CellAttention(torch.nn.Module):
    """One layer's worth: enough to make AOTI compile the cell step."""

    def forward(self, q, k, v, position):
        return torch.ops.kvcache.update_and_attend(
            q, k, v, position.reshape(-1, 1), 0, SCALE, torch.bfloat16
        )


class OffGraphKVCellCompileTest(unittest.TestCase):
    """A static one-token method and a dynamic one from two tokens both compile.

    The batching executor routes one-token steps to the first and wider ones to
    the second, so the second's width starts at 2. triton::sdpa dispatches on
    the query length (one token, 2..4 tokens, wider), and a pool past its
    split-K threshold makes those branches live under a symbolic length.
    """

    MAX_CELLS = 512
    MAX_WRITE = 32

    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_cuda()

    def test_decode_and_prefill_from_two_tokens_lower(self) -> None:
        manifest = json.dumps(
            {
                "version": 1,
                "layout": "cell",
                "maximum_capacity": self.MAX_CELLS,
                "max_cells": self.MAX_CELLS,
                "max_write": self.MAX_WRITE,
                "layers": [{"layer_id": 0, "policy": "flat"}],
            }
        ).encode()
        t = torch.export.Dim("t", min=2, max=self.MAX_WRITE)
        programs = {
            "decode": torch.export.export(
                _CellAttention(), _inputs(0, 1), strict=True
            ),
            "prefill": torch.export.export(
                _CellAttention(),
                _inputs(0, 8),
                dynamic_shapes=({2: t}, {2: t}, {2: t}, {0: t}),
                strict=True,
            ),
        }
        # The PCH path shells out to `openssl sha512` and is flaky; every
        # off-graph export turns it off, so the test compiles the same way.
        import torch._inductor.config as inductor_config

        with inductor_config.patch({"aot_inductor.precompile_headers": False}):
            lowered = self._lower(programs, manifest)
        for name in programs:
            graph = lowered.exported_program(name).graph
            self.assertTrue(
                any("executorch_call_delegate" in str(n.target) for n in graph.nodes),
                name,
            )

    @staticmethod
    def _lower(programs, manifest):
        return to_edge_transform_and_lower(
            programs,
            partitioner={
                name: [
                    CudaPartitioner(
                        [
                            CudaBackend.generate_method_name_compile_spec(name),
                            CompileSpec(OFFGRAPH_KV_COMPILE_SPEC, manifest),
                            # The step width names the delegate's input
                            # order, which partitioning decides: here the
                            # method's own (q, k, v, position), so position
                            # is input 3.
                            CompileSpec(OFFGRAPH_KV_STEP_WIDTH_COMPILE_SPEC, b"3:0"),
                            # Runtime-owned storage is declared without bytes;
                            # low-memory mode is what compiles such constants,
                            # as every off-graph export does.
                            CompileSpec("low_memory_mode", b"ON"),
                        ]
                    )
                ]
                for name in programs
            },
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
