# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.full` / `aten.full_like` export + fp64 golden for the WebGPU backend.

Both lower to the `fill` handler, which writes a constant scalar into the
pre-allocated output buffer. `full` and `full_like` are on the training-backward
critical path (gradient seeds). To keep the op in the graph (a constant-shaped
`full` would fold away), the fill size is taken from a live input: `full` uses
`x.shape`, `full_like` uses `x`. Configs span 1D/2D/4D and a non-multiple-of-256
shape that exercises the dispatch bounds guard.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

# name -> (kind, shape, fill_value)
CONFIGS = {
    "full_1d": ("full", (37,), 0.0),  # non-multiple of 256: bounds-guard path
    "full_2d": ("full", (4, 8), 3.0),
    "full_4d": ("full", (2, 3, 4, 5), -1.5),
    "full_like_2d": ("full_like", (16, 16), 7.25),
}


class FullModule(torch.nn.Module):
    def __init__(self, val: float) -> None:
        super().__init__()
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape, self.val)


class FullLikeModule(torch.nn.Module):
    def __init__(self, val: float) -> None:
        super().__init__()
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.val)


def _module(kind: str, val: float) -> torch.nn.Module:
    return (FullModule(val) if kind == "full" else FullLikeModule(val)).eval()


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


class TestFill(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (kind, shape, val) in CONFIGS.items():
            with self.subTest(name=name):
                x = torch.zeros(shape, dtype=torch.float32)
                et = _export(_module(kind, val), x)
                self.assertTrue(
                    _delegates(et), f"Expected a VulkanBackend delegate (fill {name})"
                )

    def test_golden_matches_eager(self) -> None:
        for name, (kind, shape, val) in CONFIGS.items():
            with self.subTest(name=name):
                x = torch.zeros(shape, dtype=torch.float32)
                golden = torch.full(shape, val, dtype=torch.float64)
                torch.testing.assert_close(_module(kind, val)(x).double(), golden)


if __name__ == "__main__":
    unittest.main()
