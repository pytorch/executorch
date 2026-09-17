# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.sigmoid.default` module + input for the WebGPU op-test framework.

`SigmoidModule`, `SigmoidChainedModule`, `N`, `_det_input`, and
`_wide_det_input` are imported by `cases.py` to drive the declarative op-test
suite. `SigmoidTest` is the export-delegation smoke test. Sigmoid is on the
Llama critical path (`F.silu` -> `sigmoid` + `mul`); the deterministic inputs
span the saturation tails.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# Input length; the deterministic input spans the saturation tails.
N = 64


class SigmoidModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class SigmoidChainedModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.sigmoid(x))


def _det_input() -> torch.Tensor:
    """Deterministic fp32 input spanning negatives, zero, and large magnitudes."""
    return torch.linspace(-12.0, 12.0, N, dtype=torch.float32)


def _wide_det_input() -> torch.Tensor:
    """Deterministic fp32 input covering stronger saturation tails."""
    return torch.linspace(-20.0, 20.0, N, dtype=torch.float32)


def _lower(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge, op_substr: str) -> bool:
    gm = edge.exported_program().graph_module
    return all(op_substr not in str(getattr(n, "target", "")) for n in gm.graph.nodes)


class SigmoidTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, module, x in (
            ("regular", SigmoidModule().eval(), _det_input()),
            ("chained", SigmoidChainedModule().eval(), _det_input()),
        ):
            with self.subTest(name=name):
                edge = _lower(module, x)
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et), f"Expected a VulkanBackend delegate ({name})"
                )
                self.assertTrue(
                    _op_delegated(edge, "sigmoid.default"),
                    f"sigmoid.default not delegated (fell back to CPU) for {name}",
                )
