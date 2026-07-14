# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.sum.dim_IntList` / `aten.mean.dim` single-dim reduction export + fp64 golden.

Exports single-op sum/mean graphs through VulkanPartitioner and checks the kernel
math against an fp64 torch reference. The handler reduces one dim at a time via an
outer/r/inner decomposition: `dim=-1` gives inner=1 (unit-stride reduction), a
middle dim gives inner>1 (the non-unit-stride path); `keepdim` toggles whether the
reduced dim survives in the output shape.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower


class ReduceModule(torch.nn.Module):
    def __init__(self, op: str, dim: int, keepdim: bool) -> None:
        super().__init__()
        self.op = op
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.op == "sum":
            return torch.sum(x, dim=self.dim, keepdim=self.keepdim)
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


# (name, shape, dim, keepdim): dim=-1 -> inner=1; middle dim -> inner>1.
CONFIGS = [
    ("last_dim_keep", (4, 8), -1, True),
    ("last_dim_drop", (4, 8), -1, False),
    ("middle_dim_drop", (2, 3, 4), 1, False),  # inner=4: non-unit-stride reduction
    ("middle_dim_keep", (2, 3, 4), 1, True),
]


def _det_input(shape) -> torch.Tensor:
    """Deterministic fp32 [shape]; the C++ side reconstructs it bit-for-bit.

    v[flat] = ((flat % 17) - 8) / 16 -- exact in fp32 (small modulus, po2 denominator).
    """
    n = 1
    for s in shape:
        n *= s
    flat = torch.arange(n, dtype=torch.float32)
    return ((flat % 17) - 8).div(16.0).reshape(shape)


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


def _fp64_golden(x: torch.Tensor, op: str, dim: int, keepdim: bool) -> torch.Tensor:
    xd = x.double()
    if op == "sum":
        ref = torch.sum(xd, dim=dim, keepdim=keepdim)
    else:
        ref = torch.mean(xd, dim=dim, keepdim=keepdim)
    return ref.to(torch.float32)


class TestReduce(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for op in ("sum", "mean"):
            for name, shape, dim, keepdim in CONFIGS:
                with self.subTest(op=op, config=name):
                    x = _det_input(shape)
                    et = _export(ReduceModule(op, dim, keepdim).eval(), x)
                    self.assertTrue(
                        _delegates(et),
                        f"Expected a VulkanBackend delegate ({op} {name})",
                    )

    def test_matches_fp64_golden(self) -> None:
        for op in ("sum", "mean"):
            for name, shape, dim, keepdim in CONFIGS:
                with self.subTest(op=op, config=name):
                    x = _det_input(shape)
                    got = ReduceModule(op, dim, keepdim)(x)
                    golden = _fp64_golden(x, op, dim, keepdim)
                    torch.testing.assert_close(got, golden, atol=5e-4, rtol=1e-3)


def export_reduce_model(
    op: str,
    shape,
    dim: int,
    keepdim: bool,
    pte_path: str,
    golden_path: str,
    input_path: str,
) -> None:
    """Write a reduce .pte + torch fp64 golden (raw LE fp32) + raw LE fp32 input."""
    m = ReduceModule(op, dim, keepdim).eval()
    x = _det_input(shape)
    et = _export(m, x)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    _fp64_golden(x, op, dim, keepdim).numpy().astype("<f4").tofile(golden_path)
    x.numpy().astype("<f4").tofile(input_path)
    print(f"Exported {pte_path}; golden {golden_path}; input {input_path}")


if __name__ == "__main__":
    unittest.main()
