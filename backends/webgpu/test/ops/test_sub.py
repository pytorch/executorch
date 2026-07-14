# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.sub.Tensor` export + golden for the WebGPU backend.

Exports single-op subtraction graphs through VulkanPartitioner (the WebGPU runtime
consumes the Vulkan VK00 delegate directly) and checks an fp64 torch golden
(`out = in1 - alpha * in2`). The native/etvk numeric oracle compares the GPU kernel
against this same reference; this suite locks delegation + the reference math.
Configs span same-shape 2D/3D, trailing- and leading-dim broadcast, and alpha != 1.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower


# name -> (shape_a, shape_b); b broadcasts into a.
CONFIGS = {
    "2d": ((4, 4), (4, 4)),
    "3d": ((2, 3, 4), (2, 3, 4)),
    "bcast_last": ((4, 4), (4, 1)),
    "bcast_row": ((4, 4), (1, 4)),
}


class SubModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sub(a, b)


class SubAlphaModule(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
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


def _export(m: torch.nn.Module, a: torch.Tensor, b: torch.Tensor):
    ep = torch.export.export(m.eval(), (a, b))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegates(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestSub(unittest.TestCase):
    def test_export_delegates(self) -> None:
        # Delegation => no aten.sub.Tensor left in the top-level portable graph.
        for name, (sa, sb) in CONFIGS.items():
            with self.subTest(name=name):
                a, b = _det_inputs(sa, sb)
                et = _export(SubModule(), a, b)
                self.assertTrue(
                    _delegates(et),
                    f"Expected a VulkanBackend delegate (sub {name})",
                )

    def test_golden_matches_fp64(self) -> None:
        for name, (sa, sb) in CONFIGS.items():
            with self.subTest(name=name):
                a, b = _det_inputs(sa, sb)
                ref = (a.double() - b.double()).to(torch.float32)
                torch.testing.assert_close(SubModule()(a, b), ref)

    def test_golden_matches_fp64_alpha(self) -> None:
        # Locks the handler's out = in1 - alpha * in2 path (alpha != 1).
        alpha = 2.5
        a, b = _det_inputs((4, 4), (4, 4))
        ref = (a.double() - alpha * b.double()).to(torch.float32)
        torch.testing.assert_close(SubAlphaModule(alpha)(a, b), ref)


def export_sub_model(pte_path: str, golden_path: str, input_path: str) -> None:
    """Write sub(a, b) .pte + fp64-computed torch golden + raw LE fp32 inputs (in1, in2)."""
    a, b = _det_inputs((1024, 1024), (1024, 1024))
    golden = (a.double() - b.double()).to(torch.float32).numpy().astype("<f4")
    et = _export(SubModule(), a, b)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.tofile(golden_path)
    with open(input_path, "wb") as f:
        a.numpy().astype("<f4").tofile(f)
        b.numpy().astype("<f4").tofile(f)
    print(f"Exported {pte_path}; golden {golden_path} ({golden.size} floats)")


if __name__ == "__main__":
    unittest.main()
