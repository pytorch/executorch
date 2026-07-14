# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.{eq,ne,le,ge,lt}.Scalar` export + golden for the WebGPU backend.

Each scalar comparison lowers to a single `aten.<op>.Scalar` node that the
kernel computes as `cmp(self[i], scalar)` and writes as a byte-packed bool. The
delegation test locks that every variant partitions to `VulkanBackend`; the
golden test locks the fp32 module output against the fp64 torch truth. The
deterministic ramp is exact in fp32 and straddles (and hits) the scalar, so both
precisions agree bit-for-bit and every op has mixed True/False cases; a
non-multiple-of-4 numel exercises the kernel tail (4 elems per u32 word).
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

SCALAR = 0.0
OPS = ("eq", "ne", "le", "ge", "lt")
SHAPES = {"tail": (3, 5), "3d": (2, 4, 8)}

_TORCH_OP = {
    "eq": torch.eq,
    "ne": torch.ne,
    "le": torch.le,
    "ge": torch.ge,
    "lt": torch.lt,
}


class CompareModule(torch.nn.Module):
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
        return x < self.scalar


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
        for op in OPS:
            for name, shape in SHAPES.items():
                with self.subTest(op=op, shape=name):
                    x = _det_input(shape)
                    et = _export(CompareModule(op, SCALAR).eval(), x)
                    self.assertTrue(
                        _delegated(et),
                        f"Expected a VulkanBackend delegate ({op}.Scalar {name})",
                    )

    def test_module_matches_fp64_golden(self) -> None:
        for op in OPS:
            for name, shape in SHAPES.items():
                with self.subTest(op=op, shape=name):
                    x = _det_input(shape)
                    got = CompareModule(op, SCALAR)(x)
                    golden = _TORCH_OP[op](x.double(), float(SCALAR))
                    torch.testing.assert_close(got, golden)


if __name__ == "__main__":
    unittest.main()
