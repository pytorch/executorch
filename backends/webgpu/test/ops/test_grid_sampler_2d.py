# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.grid_sampler_2d.default` module + configs for the WebGPU op-test framework.

`GridSampler2dModule` samples the input at grid coords (bilinear, border padding,
align_corners=True — the only config the backend supports). Both inputs are
float, so the op-test framework feeds both directly. `GridSampler2dTest` is the
export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, grid_shape)
CONFIGS = {
    "sq": ((1, 2, 4, 4), (1, 3, 3, 2)),
    "wide_in": ((1, 1, 3, 5), (1, 4, 4, 2)),
    "batch": ((2, 3, 4, 4), (2, 2, 6, 2)),
}


class GridSampler2dModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True
        )


def _det(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(in_shape, grid_shape):
    x = _det(in_shape, 1)
    grid = _det(grid_shape, 2)
    ep = torch.export.export(GridSampler2dModule().eval(), (x, grid))
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


class GridSampler2dTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (in_shape, grid_shape) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(in_shape, grid_shape)
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (grid_sampler {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "grid_sampler"),
                    f"grid_sampler not delegated (fell back to CPU) for {name}",
                )
