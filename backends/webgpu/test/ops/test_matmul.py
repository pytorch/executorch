# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.mm.default` (fp32 GEMM) module + configs for the WebGPU op-test framework.

`MatmulModule` + `CONFIGS` are imported by `cases.py` to drive the declarative op-test
suite (export via VulkanPartitioner + fp64 torch golden, run on Dawn). `MatmulTest` is
the export-delegation smoke test. Configs span a square matmul, a tall/skinny shape,
K==1 (rank-1 update), and non-power-of-two dims that exercise the bounds guard.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (shape_a [M, K], shape_b [K, N]). Output is [M, N].
CONFIGS = {
    "square": ((64, 64), (64, 64)),
    "tall": ((128, 32), (32, 16)),
    "k1": ((8, 1), (1, 8)),
    "nonmultiple": ((7, 5), (5, 3)),
}


class MatmulModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)


def _det_inputs(shape_a, shape_b):
    """Deterministic fp32 inputs (fixed seed) for a config."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(*shape_a, generator=g, dtype=torch.float32)
    b = torch.randn(*shape_b, generator=g, dtype=torch.float32)
    return a, b


def _export(a: torch.Tensor, b: torch.Tensor):
    ep = torch.export.export(MatmulModule().eval(), (a, b))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class MatmulTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (sa, sb) in CONFIGS.items():
            with self.subTest(name=name):
                a, b = _det_inputs(sa, sb)
                et = _export(a, b)
                self.assertTrue(
                    _delegated(et), f"Expected a VulkanBackend delegate (mm {name})"
                )
