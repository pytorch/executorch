# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.q4gsw_requant` (STE re-quant + int4 pack) export + fp32 code golden.

Writes updated fp32 latent weights back to the 4-bit group-symmetric packed codes
`et_vk.linear_q4gsw` reads (only the codes move; the per-group scale is frozen).
Reached by a direct op call in the on-device training graph after the optimizer
step. The golden is computed in fp32 (not fp64) on purpose: the kernel rounds
`round(latent / scale)` in IEEE-754 fp32, so a bit-exact contract must round in
fp32 too -- an fp64 reference would flip codes at half-way ties. CONFIGS reuse
Llama-3.2-1B linear shapes plus an odd-K shape that exercises the final-nibble
tail guard.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

# name -> (n out_features, k in_features, group_size).
CONFIGS = {
    "kv_proj": (512, 2048, 64),
    "q_proj_g32": (2048, 2048, 32),
    "small_odd_k": (6, 129, 64),  # odd K -> trailing low-nibble-only byte
}


class Q4gswRequantModule(torch.nn.Module):
    def __init__(self, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, latent: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return torch.ops.et_vk.q4gsw_requant(latent, scales, self.group_size)


def _det_inputs(n: int, k: int, gs: int):
    """Deterministic fp32 latent [N, K] + frozen scales [num_groups, N] (fixed seed).

    Scales are small relative to the latent so `latent / scale` spans well beyond
    [-8, 7], exercising the clamp on both ends.
    """
    num_groups = (k + gs - 1) // gs
    g = torch.Generator().manual_seed(0)
    latent = torch.randn(n, k, generator=g, dtype=torch.float32)
    scales = torch.rand(num_groups, n, generator=g, dtype=torch.float32) * 0.1 + 0.05
    return latent, scales


def _reference_codes(latent: torch.Tensor, scales: torch.Tensor, gs: int) -> torch.Tensor:
    """fp32 truth for the int4 codes: clamp(round(latent / scale), -8, 7), [N, K]."""
    n, k = latent.shape
    group_idx = torch.arange(k) // gs  # [K]
    scale_full = scales.t()[:, group_idx]  # [N, K]: scales[k // gs, n]
    q = torch.round(latent / scale_full)
    return torch.clamp(q, -8, 7).to(torch.int64)


def _unpack(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Undo the nibble packing: even k -> low nibble, odd k -> high nibble, code - 8."""
    n = packed.shape[0]
    p = packed.to(torch.int64)
    low = p & 0xF
    high = (p >> 4) & 0xF
    codes = torch.zeros(n, k, dtype=torch.int64)
    n_low = codes[:, 0::2].shape[1]
    n_high = codes[:, 1::2].shape[1]
    codes[:, 0::2] = low[:, :n_low] - 8
    codes[:, 1::2] = high[:, :n_high] - 8
    return codes


def _export(latent: torch.Tensor, scales: torch.Tensor, gs: int):
    ep = torch.export.export(Q4gswRequantModule(gs).eval(), (latent, scales))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestQ4gswRequant(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (n, k, gs) in CONFIGS.items():
            with self.subTest(config=name):
                latent, scales = _det_inputs(n, k, gs)
                et = _export(latent, scales, gs)
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (q4gsw_requant {name})",
                )

    def test_op_matches_fp32_golden(self) -> None:
        # Op codes vs fp32 quant truth: guards formula+layout, bit-exact.
        for name, (n, k, gs) in CONFIGS.items():
            with self.subTest(config=name):
                latent, scales = _det_inputs(n, k, gs)
                packed = torch.ops.et_vk.q4gsw_requant(latent, scales, gs)
                self.assertEqual(packed.dtype, torch.uint8)
                self.assertEqual(tuple(packed.shape), (n, (k + 1) // 2))
                got = _unpack(packed, k)
                golden = _reference_codes(latent, scales, gs)
                torch.testing.assert_close(got, golden, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
