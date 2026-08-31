# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.index_select.default` module + configs for the WebGPU op-test framework.

`IndexSelectModule` gathers rows along a dim via a baked int index buffer, so the
only runtime input is the float tensor (the op-test framework is float-only).
index_select is pure data movement (bit-identical) -> float32 oracle. `IndexSelectTest`
is the export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, dim, index)
CONFIGS = {
    "dim0_1d": ((5,), 0, [0, 2, 4, 1]),
    "dim0_2d": ((4, 7), 0, [3, 0, 1]),
    "dim1_2d": ((4, 7), 1, [5, 2, 0, 2]),
    "dim1_3d": ((3, 5, 7), 1, [4, 0, 2]),
    "dim2_3d": ((3, 5, 7), 2, [6, 1, 4]),
}


class IndexSelectModule(torch.nn.Module):
    def __init__(self, dim, index) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("index", torch.tensor(index, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, self.dim, self.index)


def _det_input(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(dim, index, x: torch.Tensor):
    ep = torch.export.export(IndexSelectModule(dim, index).eval(), (x,))
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


class IndexSelectTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, dim, index) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(dim, index, _det_input(shape))
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (index_select {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "index_select"),
                    f"index_select not delegated (fell back to CPU) for {name}",
                )
