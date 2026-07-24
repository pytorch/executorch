# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.grid_priors` module + configs for the WebGPU op-test framework.

`GridPriorsModule` calls the custom op directly (detection anchor-grid op, no
aten lowering); `stride`/`offset` are baked construct kwargs and only the float
tensor `x` is a runtime input (its VALUES are unused — only its H/W set the
output shape [H*W, 2]). The op has a CPU eager impl, so the framework goldens it
directly. `GridPriorsTest` is the export-delegation smoke test.
"""

import unittest

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, stride, offset)
CONFIGS = {
    "s8": ((1, 3, 8, 10), 8, 0.5),
    "s16": ((1, 3, 4, 4), 16, 0.0),
    "offset0": ((1, 3, 5, 7), 4, 0.0),
}


class GridPriorsModule(torch.nn.Module):
    def __init__(self, stride, offset) -> None:
        super().__init__()
        self.stride = stride
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.et_vk.grid_priors.default(x, self.stride, self.offset)


def _det(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(shape, stride, offset):
    ep = torch.export.export(GridPriorsModule(stride, offset).eval(), (_det(shape),))
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


class GridPriorsTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, stride, offset) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(shape, stride, offset)
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (grid_priors {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "grid_priors"),
                    f"grid_priors not delegated (CPU fallback) for {name}",
                )
