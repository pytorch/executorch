# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.cat.default` module + configs for the WebGPU op-test framework.

`CatModule` + `CONFIGS` are imported by `cases.py` to drive the declarative op-test
suite. `CatTest` is the export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (list_of_input_shapes, dim)
CONFIGS = {
    "dim0_3": ([(2, 3), (2, 3), (2, 3)], 0),
    "dim1_2": ([(1, 4, 8), (1, 4, 8)], 1),
    "dim2_3": ([(2, 3, 4), (2, 3, 4), (2, 3, 4)], 2),
    "uneven": ([(2, 1, 4), (2, 3, 4), (2, 2, 4)], 1),
}


class CatModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return torch.cat(xs, self.dim)


def _det_inputs(shapes):
    # Distinct value range per input so a cross-contamination bug is visible.
    inputs = []
    base = 0.0
    for sh in shapes:
        n = 1
        for s in sh:
            n *= s
        inputs.append(torch.arange(base, base + n, dtype=torch.float32).reshape(sh))
        base += 1000.0
    return tuple(inputs)


def _lower(dim, xs):
    ep = torch.export.export(CatModule(dim).eval(), xs)
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge, op_substr: str) -> bool:
    # op must be absorbed into the delegate, not left as a top-level CPU-fallback node.
    gm = edge.exported_program().graph_module
    return all(op_substr not in str(getattr(n, "target", "")) for n in gm.graph.nodes)


class CatTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shapes, dim) in CONFIGS.items():
            edge = _lower(dim, _det_inputs(shapes))
            et = edge.to_executorch()
            self.assertTrue(
                _delegated(et), f"Expected a VulkanBackend delegate (cat {name})"
            )
            self.assertTrue(
                _op_delegated(edge, "cat"),
                f"cat not delegated (fell back to CPU) for {name}",
            )
