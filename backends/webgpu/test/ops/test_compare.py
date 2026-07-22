# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Comparison-op tests for the WebGPU backend.

Two suites share this file:
- `aten.{eq,ne,le,ge,lt,gt}.Scalar` (`ScalarCompareModule` / `TestCompare`): each
  scalar comparison lowers to a single `aten.<op>.Scalar` node computed as
  `cmp(self[i], scalar)` and written as a byte-packed bool. The deterministic ramp
  is exact in fp32 and straddles (and hits) the scalar, so fp32/fp64 agree
  bit-for-bit and every op has mixed True/False cases; a non-multiple-of-4 numel
  exercises the kernel tail (4 elems per u32 word).
- `aten.{eq,lt,le,gt,ge}.Tensor` (`CompareModule` / `CompareTest`, imported by
  `cases.py`): elementwise fp32 tensor comparison -> bool. Inputs come from a small
  discrete range so `eq`/`le`/`ge` see genuine ties and `lt`/`gt` are a real
  true/false mix; numel is a multiple of 4 (bool output packs 4/word).
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

SCALAR = 0.0
SCALAR_OPS = ("eq", "ne", "le", "ge", "lt", "gt")
SCALAR_SHAPES = {"tail": (3, 5), "3d": (2, 4, 8)}

_TORCH_OP = {
    "eq": torch.eq,
    "ne": torch.ne,
    "le": torch.le,
    "ge": torch.ge,
    "lt": torch.lt,
    "gt": torch.gt,
}


class ScalarCompareModule(torch.nn.Module):
    def __init__(self, op: str, scalar: float) -> None:
        super().__init__()
        self.op = op
        self.scalar = scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.op == "eq":
            return x == self.scalar
        if self.op == "ne":
            return x != self.scalar
        if self.op == "le":
            return x <= self.scalar
        if self.op == "ge":
            return x >= self.scalar
        if self.op == "lt":
            return x < self.scalar
        return x > self.scalar


def _det_input(shape: tuple[int, ...]) -> torch.Tensor:
    """Deterministic fp32 spanning [-0.5, 0.5] in 1/16 steps (exact in fp32,
    hits and straddles 0.0)."""
    n = 1
    for d in shape:
        n *= d
    flat = torch.arange(n, dtype=torch.float32)
    return (((flat % 17) - 8) / 16.0).reshape(shape)


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestCompare(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for op in SCALAR_OPS:
            for name, shape in SCALAR_SHAPES.items():
                with self.subTest(op=op, shape=name):
                    x = _det_input(shape)
                    et = _export(ScalarCompareModule(op, SCALAR).eval(), x)
                    self.assertTrue(
                        _delegated(et),
                        f"Expected a VulkanBackend delegate ({op}.Scalar {name})",
                    )

    def test_module_matches_fp64_golden(self) -> None:
        for op in SCALAR_OPS:
            for name, shape in SCALAR_SHAPES.items():
                with self.subTest(op=op, shape=name):
                    x = _det_input(shape)
                    got = ScalarCompareModule(op, SCALAR)(x)
                    golden = _TORCH_OP[op](x.double(), float(SCALAR))
                    torch.testing.assert_close(got, golden)


class CompareModule(torch.nn.Module):
    def __init__(self, op) -> None:
        super().__init__()
        self.op = op

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.op == "eq":
            return a == b
        if self.op == "lt":
            return a < b
        if self.op == "le":
            return a <= b
        if self.op == "gt":
            return a > b
        return a >= b  # ge


def _cmp_gen(seed):
    # Small discrete range from a per-input seed: the two inputs DIFFER (a!=b)
    # while still colliding often (so eq/le/ge see genuine ties and lt/gt are a
    # real true/false mix). numel is a multiple of 4 (bool output packs 4/word).
    def g(shape):
        gen = torch.Generator().manual_seed(seed)
        return torch.randint(-2, 3, shape, generator=gen).to(torch.float32)

    return g


compare_gen_a = _cmp_gen(0)
compare_gen_b = _cmp_gen(1)


OPS = ["eq", "lt", "le", "gt", "ge"]
# All shapes have numel % 4 == 0 (bool output is packed 4 bytes/word).
SHAPES = [(4, 8), (2, 3, 8), (16, 16)]


class CompareTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for op in OPS:
            with self.subTest(op=op):
                a = compare_gen_a((4, 8))
                b = compare_gen_b((4, 8))
                ep = torch.export.export(CompareModule(op).eval(), (a, b))
                edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
                et = edge.to_executorch()
                deleg = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(deleg, f"Expected VulkanBackend delegate ({op})")


if __name__ == "__main__":
    unittest.main()
