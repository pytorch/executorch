# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.div.Tensor` (broadcast) export + fp64 golden for the WebGPU backend.

Exports single-op divide graphs through VulkanPartitioner and asserts they
delegate to the Vulkan backend (div is absent from the top-level portable ops),
then locks the golden math against an fp64 torch reference (`a / b` with PyTorch
broadcasting). Configs span the same-shape fast path, a last-dim broadcast at
LLM width, and a mixed-rank left-pad case. Divisors are bounded away from zero
so the fp32-vs-fp64 comparison stays well-conditioned.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (shape_a, shape_b). Output shape is the broadcast of the two.
CONFIGS = {
    "same": ((8, 32), (8, 32)),  # fast path (same-shape elementwise)
    "bcast_lastdim": ((1, 1, 7, 896), (1, 1, 7, 1)),  # last-dim broadcast, LLM
    "mixedrank": ((4,), (3, 4)),  # right-aligned left-pad (in.ndim < out.ndim)
}


class DivModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a / b


def _det_inputs(shape_a, shape_b):
    """Deterministic fp32 inputs (fixed seed); divisor bounded away from zero."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(*shape_a, generator=g, dtype=torch.float32)
    b = torch.randn(*shape_b, generator=g, dtype=torch.float32).abs() + 0.5
    return a, b


def _export(a: torch.Tensor, b: torch.Tensor):
    ep = torch.export.export(DivModule().eval(), (a, b))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _top_level_op_names(et) -> set[str]:
    return {
        op.name
        for plan in et.executorch_program.execution_plan
        for op in plan.operators
    }


class TestDiv(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (sa, sb) in CONFIGS.items():
            with self.subTest(name=name):
                a, b = _det_inputs(sa, sb)
                et = _export(a, b)
                self.assertTrue(
                    _delegated(et), f"Expected a VulkanBackend delegate (div {name})"
                )
                self.assertFalse(
                    any("div" in n for n in _top_level_op_names(et)),
                    f"div should be delegated, not a top-level portable op (div {name})",
                )

    def test_op_matches_fp64_golden(self) -> None:
        for name, (sa, sb) in CONFIGS.items():
            with self.subTest(name=name):
                a, b = _det_inputs(sa, sb)
                got = DivModule()(a, b)
                golden = (a.double() / b.double()).to(torch.float32)
                torch.testing.assert_close(got, golden, atol=5e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
