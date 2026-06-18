# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.permute_copy.default` module + configs for the WebGPU op-test framework.

`PermuteModule` + `CONFIGS` are imported by `cases.py` to drive the declarative
op-test suite. `PermuteTest` is the export-delegation smoke
test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, perm)
CONFIGS = {
    "rot3d": ((2, 3, 4), (2, 0, 1)),
    "mid4d": ((1, 8, 4, 16), (0, 2, 1, 3)),
    "t2d": ((3, 5), (1, 0)),
    "shuffle4d": ((2, 3, 4, 5), (3, 1, 0, 2)),
}


class PermuteModule(torch.nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.permute(x, self.perm).contiguous()


def _det_input(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(perm, x: torch.Tensor):
    ep = torch.export.export(PermuteModule(perm).eval(), (x,))
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


class PermuteTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, perm) in CONFIGS.items():
            edge = _lower(perm, _det_input(shape))
            et = edge.to_executorch()
            self.assertTrue(
                _delegated(et), f"Expected a VulkanBackend delegate (permute {name})"
            )
            self.assertTrue(
                _op_delegated(edge, "permute"),
                f"permute not delegated (fell back to CPU) for {name}",
            )
