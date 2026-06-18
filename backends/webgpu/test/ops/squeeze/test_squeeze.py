# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.squeeze_copy.dims` module + configs for the WebGPU op-test framework.

`SqueezeModule` + `CONFIGS` are imported by `cases.py` to drive the declarative
op-test suite. `SqueezeTest` is the export-delegation smoke
test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, squeeze_dim)
CONFIGS = {
    "dim0": ((1, 3, 4), 0),
    "mid": ((2, 1, 4), 1),
    "multi": ((1, 3, 1, 4), (0, 2)),
}


class SqueezeModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(x, self.dim)


def _det_input(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(dim, x: torch.Tensor):
    ep = torch.export.export(SqueezeModule(dim).eval(), (x,))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge, op_substr: str) -> bool:
    # op must be absorbed into the delegate, not left as a CPU-fallback node.
    gm = edge.exported_program().graph_module
    return all(op_substr not in str(getattr(n, "target", "")) for n in gm.graph.nodes)


class SqueezeTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, dim) in CONFIGS.items():
            edge = _lower(dim, _det_input(shape))
            et = edge.to_executorch()
            self.assertTrue(
                _delegated(et),
                f"Expected a VulkanBackend delegate (squeeze {name})",
            )
            self.assertTrue(
                _op_delegated(edge, "squeeze_copy"),
                f"squeeze_copy not delegated (fell back to CPU) for {name}",
            )
