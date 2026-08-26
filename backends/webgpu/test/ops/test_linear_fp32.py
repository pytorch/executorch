# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.linear.default` (fp32) module + inputs for the WebGPU op-test framework.

`make_linear` and `_ramp` are imported by `cases.py` to drive the declarative
op-test suite. `LinearFp32Test` is the export-delegation smoke test. fp32 linear
is the projection used throughout BART + the DaViT vision encoder (Florence-2);
both the bias and no-bias paths are covered.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class LinearFp32Module(torch.nn.Module):
    """fp32 linear; lowers to aten.linear.default with a prepacked [N, K] weight."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def make_linear(
    in_features: int, out_features: int, bias: bool = True
) -> torch.nn.Module:
    """Factory with deterministic weight (+ bias): a normalized ramp per row."""
    m = LinearFp32Module(in_features, out_features, bias=bias)
    with torch.no_grad():
        w = torch.linspace(
            -1.0, 1.0, out_features * in_features, dtype=torch.float32
        ).reshape(out_features, in_features)
        m.fc.weight.copy_(w / in_features)
        if bias:
            m.fc.bias.copy_(
                torch.linspace(-0.5, 0.5, out_features, dtype=torch.float32)
            )
    return m


def _ramp(shape) -> torch.Tensor:
    """Deterministic linear ramp in [-1, 1] reshaped to `shape`."""
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(shape)


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class LinearFp32Test(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for bias in (True, False):
            et = _export(make_linear(64, 32, bias=bias).eval(), _ramp((4, 64)))
            found = any(
                d.id == "VulkanBackend"
                for plan in et.executorch_program.execution_plan
                for d in plan.delegates
            )
            self.assertTrue(
                found, f"Expected a VulkanBackend delegate (linear bias={bias})"
            )
