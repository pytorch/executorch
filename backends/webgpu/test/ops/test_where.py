# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.where.self` export + fp64 golden for the WebGPU backend.

`where(cond, a, b) -> cond ? a : b`, with cond a 1-byte bool and a/b fp32
(broadcast across all three operands). The kernel reads cond byte-packed as
`array<u32>` and relinearizes each out coord onto every operand. Configs cover
the equal-shape path plus broadcasts that exercise the size-1 clamp on cond, a,
and b. The native binary has no ATen, so the golden is computed with torch here
and checked in etvk CI.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (cond_shape, a_shape, b_shape). Output is the broadcast of all three.
CONFIGS = {
    "equal": ((4, 8), (4, 8), (4, 8)),
    "broadcast": ((4, 1), (4, 8), (1, 8)),
    "cond_row": ((8,), (4, 8), (4, 8)),
}


class WhereModule(torch.nn.Module):
    def forward(
        self, cond: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(cond, a, b)


def _det_inputs(cond_shape, a_shape, b_shape):
    """Deterministic (bool cond, fp32 a, fp32 b) for a config."""
    g = torch.Generator().manual_seed(0)
    cond = torch.rand(cond_shape, generator=g) > 0.5
    a = torch.randn(*a_shape, generator=g, dtype=torch.float32)
    b = torch.randn(*b_shape, generator=g, dtype=torch.float32)
    return cond, a, b


def _fp64_golden(cond, a, b):
    return torch.where(cond, a.double(), b.double()).to(torch.float32)


def _export(cond, a, b):
    ep = torch.export.export(WhereModule().eval(), (cond, a, b))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegates(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestWhere(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (cs, as_, bs) in CONFIGS.items():
            with self.subTest(config=name):
                cond, a, b = _det_inputs(cs, as_, bs)
                et = _export(cond, a, b)
                self.assertTrue(
                    _delegates(et), f"Expected a VulkanBackend delegate (where {name})"
                )

    def test_op_matches_fp64_golden(self) -> None:
        for name, (cs, as_, bs) in CONFIGS.items():
            with self.subTest(config=name):
                cond, a, b = _det_inputs(cs, as_, bs)
                out = WhereModule()(cond, a, b)
                torch.testing.assert_close(out, _fp64_golden(cond, a, b))


if __name__ == "__main__":
    unittest.main()
