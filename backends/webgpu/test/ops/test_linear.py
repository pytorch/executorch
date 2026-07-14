# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.linear.default` (fp32, no bias) export + golden for the WebGPU backend.

Exports single-op linear graphs through VulkanPartitioner and locks the
delegation contract + an fp64 torch golden. The handler computes
out[m,n] = sum_k x[m,k] * w[n,k] (weight is [N,K]); it picks a vec4-over-K
kernel when K % 4 == 0 and a scalar tiled kernel otherwise, so the configs span
both branches (plus a K==1 / non-multiple shape that exercises the bounds guard).
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

# name -> (M, K, N). weight is [N, K]; output is [M, N].
CONFIGS = {
    "square": (64, 64, 64),  # K % 4 == 0 -> vec4-over-K kernel
    "tall_vec4": (32, 128, 16),  # K % 4 == 0 -> vec4 path, skinny N
    "scalar_k": (8, 7, 5),  # K % 4 != 0 -> scalar tiled path + bounds guard
    "k1": (4, 1, 8),  # K == 1 rank-1, scalar path
}


class LinearModule(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        n, k = weight.shape
        self.linear = torch.nn.Linear(k, n, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _det_inputs(m: int, k: int, n: int):
    """Deterministic fp32 input [M,K] + weight [N,K] (fixed seed)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(m, k, generator=g, dtype=torch.float32)
    w = torch.randn(n, k, generator=g, dtype=torch.float32)
    return x, w


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegates(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestLinear(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (m, k, n) in CONFIGS.items():
            with self.subTest(name=name):
                x, w = _det_inputs(m, k, n)
                et = _export(LinearModule(w).eval(), x)
                self.assertTrue(
                    _delegates(et), f"Expected a VulkanBackend delegate (linear {name})"
                )

    def test_golden_matches_fp64(self) -> None:
        # Golden must match the fp64 F.linear truth (weight [N,K], no bias).
        for name, (m, k, n) in CONFIGS.items():
            with self.subTest(name=name):
                x, w = _det_inputs(m, k, n)
                got = LinearModule(w).eval()(x)
                golden = torch.nn.functional.linear(x.double(), w.double())
                torch.testing.assert_close(
                    got, golden.to(torch.float32), atol=1e-3, rtol=1e-3
                )


def export_linear_model(
    pte_path: str, golden_path: str, input_path: str, config: str = "square"
) -> None:
    """Write a linear .pte + torch fp64 golden (raw LE fp32) + raw LE fp32 input."""
    m_dim, k, n = CONFIGS[config]
    x, w = _det_inputs(m_dim, k, n)
    model = LinearModule(w).eval()
    golden = (
        torch.nn.functional.linear(x.double(), w.double())
        .to(torch.float32)
        .numpy()
        .astype("<f4")
    )
    et = _export(model, x)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.tofile(golden_path)
    x.numpy().astype("<f4").tofile(input_path)
    print(
        f"Exported {pte_path}; golden {golden_path} ({golden.size} floats); "
        f"input {input_path} ({x.numel()} floats)"
    )


if __name__ == "__main__":
    unittest.main()
