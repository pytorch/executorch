# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.sub.Tensor` (broadcast + alpha) module + configs for the WebGPU op-test framework.

`SubModule` + `CONFIGS` are imported by `cases.py` to drive the declarative op-test
suite (export via VulkanPartitioner + fp64 torch golden, run on Dawn). `SubTest` is
the export-delegation smoke test. Configs span the same-shape fast path, the
middle/spatial broadcast `[N,C,H,W] - [N,C,1,1]` (InstanceNorm `x - mean`), and an
alpha != 1 case (`in1 - alpha * in2`).
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (shape_a, shape_b, alpha). Output shape is the broadcast of the two.
CONFIGS = {
    "same": ((8, 32), (8, 32), 1.0),  # fast path (same-shape)
    "bcast_spatial": ((1, 8, 16, 16), (1, 8, 1, 1), 1.0),  # InstanceNorm x-mean [N,C,1,1]
    "alpha": ((8, 32), (8, 32), 2.0),  # alpha != 1 (in1 - alpha * in2)
}


class SubModule(torch.nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sub(a, b, alpha=self.alpha)


def _det_inputs(shape_a, shape_b):
    """Deterministic fp32 inputs (fixed seed) for a config."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(*shape_a, generator=g, dtype=torch.float32)
    b = torch.randn(*shape_b, generator=g, dtype=torch.float32)
    return a, b


def _export(a: torch.Tensor, b: torch.Tensor, alpha: float):
    ep = torch.export.export(SubModule(alpha).eval(), (a, b))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class SubTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (sa, sb, alpha) in CONFIGS.items():
            with self.subTest(name=name):
                a, b = _det_inputs(sa, sb)
                et = _export(a, b, alpha)
                self.assertTrue(
                    _delegated(et), f"Expected a VulkanBackend delegate (sub {name})"
                )
