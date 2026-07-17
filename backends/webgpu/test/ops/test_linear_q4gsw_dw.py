# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.linear_q4gsw_dw` (STE weight gradient) export + fp64 golden.

The weight gradient of a frozen 4-bit linear: `d_W[N, K] = d_out^T @ x`, the grad
wrt the dequantized weight (both operands are fp32; no int4 unpack). Reached by a
direct op call in the on-device training graph. CONFIGS reuse Llama-3.2-1B linear
shapes plus a non-tile-aligned shape that exercises the kernel's boundary clamp.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (m tokens, k in_features, n out_features).
CONFIGS = {
    "q_proj_112": (112, 2048, 2048),
    "kv_proj_112": (112, 2048, 512),
    "boundary": (13, 18, 10),  # non-multiple-of-4: exercises the min()-clamp
}


class Q4gswDwModule(torch.nn.Module):
    def forward(self, d_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.et_vk.linear_q4gsw_dw(d_out, x)


def _det_inputs(m: int, k: int, n: int):
    """Deterministic fp32 d_out [m, n] + x [m, k] (fixed seed)."""
    g = torch.Generator().manual_seed(0)
    d_out = torch.randn(m, n, generator=g, dtype=torch.float32)
    x = torch.randn(m, k, generator=g, dtype=torch.float32)
    return d_out, x


def _fp64_golden(d_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """fp64 truth: d_W = d_out^T @ x, [N, M] @ [M, K] = [N, K]."""
    return (d_out.double().t() @ x.double()).to(torch.float32)


def _export(d_out: torch.Tensor, x: torch.Tensor):
    ep = torch.export.export(Q4gswDwModule().eval(), (d_out, x))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestLinearQ4gswDw(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (m, k, n) in CONFIGS.items():
            with self.subTest(config=name):
                d_out, x = _det_inputs(m, k, n)
                et = _export(d_out, x)
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (linear_q4gsw_dw {name})",
                )

    def test_op_matches_fp64_golden(self) -> None:
        # Op (d_out^T @ x) vs fp64 matmul truth: guards the formula.
        for name, (m, k, n) in CONFIGS.items():
            with self.subTest(config=name):
                d_out, x = _det_inputs(m, k, n)
                got = torch.ops.et_vk.linear_q4gsw_dw(d_out, x)
                golden = _fp64_golden(d_out, x)
                self.assertEqual(tuple(got.shape), (n, k))
                torch.testing.assert_close(got, golden, atol=5e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
