# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.bmm.default` (fp32 batched GEMM) export + golden for the WebGPU backend.

Exports single-op batched-matmul graphs through VulkanPartitioner and writes a
torch-computed fp64 golden (the native binary has no ATen) + the raw fp32 inputs
the native test loads and compares. Configs span the vec4 fast path (K%4==0 and
N%4==0, wider 16B loads) and the scalar tiled path (K or N not a multiple of 4,
which also exercises the bounds guards), across batch sizes 1, 2, and 4.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

# name -> (shape_a [B, M, K], shape_b [B, K, N]). Output is [B, M, N].
CONFIGS = {
    "square": ((2, 64, 64), (2, 64, 64)),  # vec4 path, multi-tile, B=2
    "tall": ((4, 128, 16), (4, 16, 24)),  # vec4 path, tall M, B=4
    "scalar": ((2, 7, 5), (2, 5, 3)),  # scalar tiled path (K,N %4!=0)
    "batch1": ((1, 33, 17), (1, 17, 5)),  # B=1, scalar path, bounds guards
}


class BmmModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


def _det_inputs(shape_a, shape_b):
    """Deterministic fp32 inputs (fixed seed) for a config."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(*shape_a, generator=g, dtype=torch.float32)
    b = torch.randn(*shape_b, generator=g, dtype=torch.float32)
    return a, b


def _fp64_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp64 truth: bmm(a, b) accumulated in double, cast back to fp32."""
    return torch.bmm(a.double(), b.double()).to(torch.float32)


def _export(a: torch.Tensor, b: torch.Tensor):
    ep = torch.export.export(BmmModule().eval(), (a, b))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class BmmTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (sa, sb) in CONFIGS.items():
            with self.subTest(name=name):
                a, b = _det_inputs(sa, sb)
                et = _export(a, b)
                self.assertTrue(
                    _delegated(et), f"Expected a VulkanBackend delegate (bmm {name})"
                )

    def test_golden_matches_fp64(self) -> None:
        for name, (sa, sb) in CONFIGS.items():
            with self.subTest(name=name):
                a, b = _det_inputs(sa, sb)
                torch.testing.assert_close(
                    BmmModule()(a, b), _fp64_golden(a, b), atol=1e-4, rtol=1e-3
                )


def export_bmm_model(pte_path: str, golden_path: str, input_path: str) -> None:
    """Write a bmm .pte + torch fp64 golden (raw LE fp32) + raw LE fp32 inputs (a then b)."""
    a, b = _det_inputs(*CONFIGS["square"])
    et = _export(a, b)
    golden = _fp64_golden(a, b).numpy().astype("<f4")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.tofile(golden_path)
    with open(input_path, "wb") as f:
        f.write(a.numpy().astype("<f4").tobytes())
        f.write(b.numpy().astype("<f4").tobytes())
    print(
        f"Exported {pte_path}; golden {golden_path} ({golden.size} floats); "
        f"inputs {input_path} ({a.numel() + b.numel()} floats)"
    )


if __name__ == "__main__":
    unittest.main()
