# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.mul.Tensor` (full broadcast) module + configs for the WebGPU op-test framework.

`MulModule` + `CONFIGS` are imported by `cases.py` to drive the declarative op-test
suite (export via VulkanPartitioner + fp64 torch golden, run on Dawn). `MulTest` is
the export-delegation smoke test. Configs span the same-shape
fast path (SwiGLU), last-dim broadcast at LLM width, and a mixed-rank left-pad case.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (shape_a, shape_b). Output shape is the broadcast of the two.
CONFIGS = {
    "same": ((8, 32), (8, 32)),  # fast path (SwiGLU same-shape)
    "bcast_lastdim": ((1, 1, 7, 896), (1, 1, 7, 1)),  # last-dim broadcast, LLM width
    "mixedrank": ((4,), (3, 4)),  # right-aligned left-pad (in.ndim < out.ndim)
}


class MulModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


def _det_inputs(shape_a, shape_b):
    """Deterministic fp32 inputs (fixed seed) for a config."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(*shape_a, generator=g, dtype=torch.float32)
    b = torch.randn(*shape_b, generator=g, dtype=torch.float32)
    return a, b


def _export(a: torch.Tensor, b: torch.Tensor):
    ep = torch.export.export(MulModule().eval(), (a, b))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class MulTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (sa, sb) in CONFIGS.items():
            a, b = _det_inputs(sa, sb)
            et = _export(a, b)
            self.assertTrue(
                _delegated(et), f"Expected a VulkanBackend delegate (mul {name})"
            )
