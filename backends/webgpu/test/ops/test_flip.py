# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.flip.default` module + configs for the WebGPU op-test framework.

`FlipModule` reverses the given dims. flip is pure data movement (bit-identical),
so the suite uses the float32 oracle. `FlipTest` is the export-delegation smoke
test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, dims)
CONFIGS = {
    "mid_3d": ((2, 3, 4), [1]),
    "multi_4d": ((2, 3, 4, 5), [1, 3]),
    "last_2d": ((3, 5), [-1]),
    "all_3d": ((2, 3, 4), [0, 1, 2]),
}


class FlipModule(torch.nn.Module):
    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, self.dims)


def _det_input(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(dims, x: torch.Tensor):
    ep = torch.export.export(FlipModule(dims).eval(), (x,))
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


class FlipTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, dims) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(dims, _det_input(shape))
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (flip {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "flip"),
                    f"flip not delegated (fell back to CPU) for {name}",
                )
