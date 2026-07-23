# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.logical_not.default` export + golden for the WebGPU backend.

Exports single-op logical_not graphs through VulkanPartitioner and writes a
torch-computed golden. logical_not lowers to a byte-packed bool kernel (1 byte /
elem, one u32 word / thread), so the golden is an EXACT bool match -- the native
test compares the raw uint8 output byte-for-byte. The bool input is produced by a
delegated `x >= threshold` compare; the golden uses the De Morgan inverse
`x < threshold` as an independent reference. A non-multiple-of-4 numel exercises
the shader's partial-final-word masking (the `i < num_elements` branch).
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> shape. "tail3x7" has numel 21 (not a multiple of 4) => partial word.
CONFIGS = {
    "vec1d": (16,),
    "mat2d": (4, 8),
    "tail3x7": (3, 7),
    "cube3d": (2, 4, 8),
}


class LogicalNotModule(torch.nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(x >= self.threshold)


def _det_input(shape: tuple[int, ...]) -> torch.Tensor:
    """Deterministic fp32 spanning negatives, zero, and positives so the bool
    input to logical_not carries both True and False values."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegates(edge) -> bool:
    et = edge.to_executorch()
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _top_level_targets(edge) -> list[str]:
    return [
        str(n.target)
        for n in edge.exported_program().graph_module.graph.nodes
        if n.op == "call_function"
    ]


class TestLogicalNot(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, shape in CONFIGS.items():
            with self.subTest(config=name):
                edge = _lower(LogicalNotModule(0.0).eval(), _det_input(shape))
                targets = _top_level_targets(edge)
                self.assertFalse(
                    any("logical_not" in t for t in targets),
                    f"logical_not must be delegated, not portable ({name}): {targets}",
                )
                self.assertTrue(
                    _delegates(edge),
                    f"Expected a VulkanBackend delegate (logical_not {name})",
                )

    def test_golden_matches_eager(self) -> None:
        for name, shape in CONFIGS.items():
            with self.subTest(config=name):
                x = _det_input(shape)
                got = LogicalNotModule(0.0)(x)
                golden = x < 0.0  # De Morgan inverse of logical_not(x >= 0)
                self.assertEqual(got.dtype, torch.bool)
                torch.testing.assert_close(got, golden)


def export_logical_not_model(pte_path: str, golden_path: str, input_path: str) -> None:
    """Write logical_not .pte + torch golden (raw uint8) + raw LE fp32 input."""
    m = LogicalNotModule(0.0).eval()
    x = _det_input(CONFIGS["mat2d"])
    golden = m(x).detach().numpy().astype("<u1")
    et = _lower(m, x).to_executorch()
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.tofile(golden_path)
    x.numpy().astype("<f4").tofile(input_path)
    print(
        f"Exported {pte_path}; golden {golden_path} ({golden.size} bytes); "
        f"input {input_path} ({x.numel()} floats)"
    )


if __name__ == "__main__":
    unittest.main()
