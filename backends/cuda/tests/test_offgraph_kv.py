# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

# The very functions the lowering pass traces and splices into the graph, so a
# passing test cannot agree with a decomposition the pass does not emit.
from executorch.backends.cuda.passes.lower_offgraph_kv import (
    flat_step,
    ring_attention_mask,
    ring_physical_capacity,
    ring_step,
)
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
    # Declared at full capacity: index_copy_ and sdpa both address through
    # these strides, which is what the runtime allocation has to match.
    shape = (1, N_KV_HEADS, capacity, HEAD_DIM)
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
            out = flat_step(q, k, v, position, k_storage, v_storage, SCALE)
            self._assert_matches(out, oracle, q, k, v, position)

    def test_flat_write_lands_at_logical_position(self) -> None:
        torch.manual_seed(1)
        k_storage, v_storage = _storage(64)
        q, k, v, position = _inputs(8, 8)

        flat_step(q, k, v, position, k_storage, v_storage, SCALE)

        self.assertTrue(torch.equal(k_storage[:, :, 8:16], k))
        self.assertTrue(torch.equal(v_storage[:, :, 8:16], v))
        # Untouched slots stay untouched -- the declared stride is the full
        # capacity, so a stride mistake would scatter the write.
        self.assertFalse(k_storage[:, :, 16:].any())

    def test_flat_decode_over_large_buffer_matches_oracle(self) -> None:
        # Past _SPLITK_LKV_THRESHOLD a decode with kv_len routes through the
        # split-K path inside sdpa, which is a different kernel.
        torch.manual_seed(2)
        capacity = 512
        k_storage, v_storage = _storage(capacity)
        oracle = _Oracle(capacity)
        self.addCleanup(oracle.close)

        q, k, v, position = _inputs(0, 257)
        flat_step(q, k, v, position, k_storage, v_storage, SCALE)
        oracle.step(q, k, v, position)

        q, k, v, position = _inputs(257, 1)
        out = flat_step(q, k, v, position, k_storage, v_storage, SCALE)
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
            out = ring_step(
                q, k, v, position, k_storage, v_storage, mask, SCALE, buf_size
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
            expected = [
                0 <= query_pos - p < window for p in held  # causal + sliding
            ]
            self.assertEqual(mask[0, 0, row].tolist(), expected)
