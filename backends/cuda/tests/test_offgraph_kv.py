# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.cuda.triton.kernels.offgraph_kv import (
    cuda_offgraph_update_and_attend,
    ring_physical_capacity,
)
from executorch.extension.llm.cache.reference_cache import (
    CacheConfig,
    LayerPolicy,
    SequenceReferenceCache,
)

# Importing the op module registers kvcache::update_and_attend and exposes the
# registry the eager implementation reads its cache from.
from executorch.extension.llm.cache.update_and_attend import REGISTRY


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        raise unittest.SkipTest("BF16 not supported")


class _Oracle:
    """``kvcache::update_and_attend`` itself, stepped alongside the kernel.

    The op is the contract every backend implements, so driving it here checks
    the Triton kernels against the same placement and masking rules the other
    backends are checked against -- rather than against a mask rebuilt in this
    file, which can only ever agree with itself.

    It is stateful in the same way the kernels are: feed it each step's k/v and
    it keeps the history, so callers do not accumulate one.
    """

    _SCALE = 0.125

    def __init__(self, capacity: int, window: int = 0) -> None:
        self._cache = SequenceReferenceCache(
            CacheConfig(
                n_layers=1,
                n_kv_heads=2,
                head_dim=64,
                capacity=capacity,
                dtype=torch.float32,
                layers=(LayerPolicy.ring(window) if window else LayerPolicy.flat(),),
            )
        )
        self._key = f"cuda-offgraph-test-{id(self)}"
        REGISTRY.install(self._key, self._cache)

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        # The reference keeps its history on CPU in float32; position is
        # [q_len, n_dims] per the op contract.
        with REGISTRY.active(self._key):
            return torch.ops.kvcache.update_and_attend(
                q.float().cpu(),
                k.float().cpu(),
                v.float().cpu(),
                position.reshape(-1, 1).cpu(),
                0,
                self._SCALE,
                torch.float32,
            )

    def close(self) -> None:
        REGISTRY.uninstall(self._key)


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float().cpu() - expected).abs().max().item()


class OffGraphKVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _skip_if_no_cuda()
        cls.op = cuda_offgraph_update_and_attend

    def _step(
        self,
        storage: tuple[torch.Tensor, torch.Tensor],
        capacity: torch.Tensor,
        start: int,
        length: int,
        policy: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(1, 4, length, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 2, length, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        position = torch.arange(start, start + length, device="cuda")
        out = self.op(
            q,
            k,
            v,
            position,
            storage[0],
            storage[1],
            capacity,
            policy,
            window,
            0.125,
            torch.bfloat16,
        )
        return out, q, k, v

    def test_flat_append_and_gqa_attention(self) -> None:
        torch.manual_seed(0)
        capacity = 64
        storage = (
            torch.zeros(1, 2, capacity, 64, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, 2, capacity, 64, device="cuda", dtype=torch.bfloat16),
        )
        capacity_tensor = torch.tensor([capacity], device="cuda")
        oracle = _Oracle(capacity)
        self.addCleanup(oracle.close)

        for start in (0, 8):
            out, q, k, v = self._step(
                storage, capacity_tensor, start, 8, policy=0, window=0
            )
            position = torch.arange(start, start + 8, device="cuda")
            expected = oracle.step(q, k, v, position)
            self.assertLess(_max_abs_diff(out, expected), 1e-2)
            self.assertTrue(torch.equal(storage[0][:, :, start : start + 8], k))
            self.assertTrue(torch.equal(storage[1][:, :, start : start + 8], v))

    def test_flat_write_at_capacity_boundary(self) -> None:
        capacity = torch.tensor([8], device="cuda")
        storage = (
            torch.zeros(1, 2, 8, 64, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, 2, 8, 64, device="cuda", dtype=torch.bfloat16),
        )

        out, _, k, v = self._step(storage, capacity, 4, 4, policy=0, window=0)

        self.assertEqual(out.shape, (1, 4, 4, 64))
        self.assertTrue(torch.equal(storage[0][:, :, 4:], k))
        self.assertTrue(torch.equal(storage[1][:, :, 4:], v))

    def test_flat_split_k_decode_matches_reference(self) -> None:
        torch.manual_seed(2)
        capacity_value = 512
        storage = (
            torch.zeros(
                1, 2, capacity_value, 64, device="cuda", dtype=torch.bfloat16
            ),
            torch.zeros(
                1, 2, capacity_value, 64, device="cuda", dtype=torch.bfloat16
            ),
        )
        capacity = torch.tensor([capacity_value], device="cuda")
        oracle = _Oracle(capacity_value)
        self.addCleanup(oracle.close)
        _, prefill_q, prefill_k, prefill_v = self._step(
            storage, capacity, 0, 257, policy=0, window=0
        )
        oracle.step(
            prefill_q,
            prefill_k,
            prefill_v,
            torch.arange(0, 257, device="cuda"),
        )
        out, q, k, v = self._step(
            storage, capacity, 257, 1, policy=0, window=0
        )
        expected = oracle.step(
            q, k, v, torch.tensor([257], device="cuda")
        )
        self.assertLess(_max_abs_diff(out, expected), 1e-2)

    def test_flat_rejects_write_past_capacity(self) -> None:
        capacity = torch.tensor([8], device="cuda")
        storage = (
            torch.zeros(1, 2, 8, 64, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, 2, 8, 64, device="cuda", dtype=torch.bfloat16),
        )

        with self.assertRaisesRegex(RuntimeError, "exceeds physical capacity"):
            self._step(storage, capacity, 6, 4, policy=0, window=0)

    def test_ring_wrap_preserves_logical_attention_order(self) -> None:
        torch.manual_seed(1)
        physical_capacity = 32
        window = 16
        storage = (
            torch.zeros(
                1, 2, physical_capacity, 64, device="cuda", dtype=torch.bfloat16
            ),
            torch.zeros(
                1, 2, physical_capacity, 64, device="cuda", dtype=torch.bfloat16
            ),
        )
        capacity = torch.tensor([physical_capacity], device="cuda")
        # The oracle bounds the logical sequence, not the ring's slots.
        oracle = _Oracle(capacity=1024, window=window)
        self.addCleanup(oracle.close)

        for start, length in ((0, 24), (24, 16)):
            out, q, k, v = self._step(
                storage, capacity, start, length, policy=1, window=window
            )
            position = torch.arange(start, start + length, device="cuda")
            expected = oracle.step(q, k, v, position)
            self.assertLess(_max_abs_diff(out, expected), 1e-2)

    def test_ring_serves_a_max_write_step_mid_sequence(self) -> None:
        # A step of max_write tokens starting past the window is the chunked
        # prefill case: its earliest query reads back window-1 positions before
        # the step, so those must survive the step's own writes. A ring sized to
        # 2*window instead of ring_physical_capacity() overwrites them.
        torch.manual_seed(2)
        window = 16
        max_write = 2 * window
        physical_capacity = ring_physical_capacity(window, max_write)
        storage = (
            torch.zeros(
                1, 2, physical_capacity, 64, device="cuda", dtype=torch.bfloat16
            ),
            torch.zeros(
                1, 2, physical_capacity, 64, device="cuda", dtype=torch.bfloat16
            ),
        )
        capacity = torch.tensor([physical_capacity], device="cuda")
        oracle = _Oracle(capacity=1024, window=window)
        self.addCleanup(oracle.close)

        for start in (0, max_write, 2 * max_write):
            out, q, k, v = self._step(
                storage, capacity, start, max_write, policy=1, window=window
            )
            position = torch.arange(start, start + max_write, device="cuda")
            expected = oracle.step(q, k, v, position)
            self.assertLess(_max_abs_diff(out, expected), 1e-2)
