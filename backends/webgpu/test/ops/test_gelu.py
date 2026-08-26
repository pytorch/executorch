# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.gelu.default` module + input for the WebGPU op-test framework.

`GeluModule`, `N`, and `_det_input` are imported by `cases.py` to drive the
declarative op-test suite. `GeluTest` is the export-delegation smoke test. The
`approximate` kwarg selects the exact (erf) path ("none", PyTorch's default and
the Florence-2/BART path) or the tanh approximation; the deterministic input
spans negatives, zero, and the saturation region.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# Input length; the deterministic input spans negatives, zero, and positives.
N = 64


class GeluModule(torch.nn.Module):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate=self.approximate)


def _det_input() -> torch.Tensor:
    """Deterministic fp32 input spanning negatives, zero, and positives."""
    return torch.linspace(-6.0, 6.0, N, dtype=torch.float32)


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class GeluTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for approximate in ("none", "tanh"):
            et = _export(GeluModule(approximate).eval(), _det_input())
            found = any(
                d.id == "VulkanBackend"
                for plan in et.executorch_program.execution_plan
                for d in plan.delegates
            )
            self.assertTrue(
                found, f"Expected a VulkanBackend delegate (gelu {approximate})"
            )
