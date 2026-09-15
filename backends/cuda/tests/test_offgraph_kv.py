# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F
from executorch.backends.cuda.triton.kernels.offgraph_kv import (
    cuda_offgraph_update_and_attend,
)


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        raise unittest.SkipTest("BF16 not supported")


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position: torch.Tensor,
    window: int = 0,
) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    keys = torch.arange(k.shape[2], device=q.device)
    mask = keys.unsqueeze(0) <= position.unsqueeze(1)
    if window:
        mask &= position.unsqueeze(1) - keys.unsqueeze(0) < window
    return F.scaled_dot_product_attention(
        q.float(),
        k.repeat_interleave(groups, dim=1).float(),
        v.repeat_interleave(groups, dim=1).float(),
        attn_mask=mask.unsqueeze(0).unsqueeze(0),
        scale=0.125,
    )


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
        history_k = []
        history_v = []

        for start in (0, 8):
            out, q, k, v = self._step(
                storage, capacity_tensor, start, 8, policy=0, window=0
            )
            history_k.append(k)
            history_v.append(v)
            position = torch.arange(start, start + 8, device="cuda")
            expected = _reference(
                q,
                torch.cat(history_k, dim=2),
                torch.cat(history_v, dim=2),
                position,
            )
            self.assertLess((out.float() - expected).abs().max().item(), 1e-2)
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
        _, _, history_k, history_v = self._step(
            storage, capacity, 0, 257, policy=0, window=0
        )
        out, q, k, v = self._step(
            storage, capacity, 257, 1, policy=0, window=0
        )
        expected = _reference(
            q,
            torch.cat((history_k, k), dim=2),
            torch.cat((history_v, v), dim=2),
            torch.tensor([257], device="cuda"),
        )
        self.assertLess((out.float() - expected).abs().max().item(), 1e-2)

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
        history_k = []
        history_v = []

        for start, length in ((0, 24), (24, 16)):
            out, q, k, v = self._step(
                storage, capacity, start, length, policy=1, window=window
            )
            history_k.append(k)
            history_v.append(v)
            position = torch.arange(start, start + length, device="cuda")
            expected = _reference(
                q,
                torch.cat(history_k, dim=2),
                torch.cat(history_v, dim=2),
                position,
                window,
            )
            self.assertLess((out.float() - expected).abs().max().item(), 1e-2)
