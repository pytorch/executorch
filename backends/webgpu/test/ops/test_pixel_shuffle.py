# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.pixel_shuffle.default` module + configs for the WebGPU op-test framework.

`PixelShuffleModule` rearranges (N, C*r*r, H, W) -> (N, C, H*r, W*r). pixel_shuffle
is pure data movement (bit-identical), so the suite uses the float32 oracle.
`PixelShuffleTest` is the export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> input_shape, upscale_factor
CONFIGS = {
    "r2": ((1, 8, 2, 3), 2),
    "r2_batch": ((2, 4, 3, 3), 2),
    "r3": ((1, 9, 2, 2), 3),
    "r2_3d": ((4, 2, 2), 2),
}


class PixelShuffleModule(torch.nn.Module):
    def __init__(self, r) -> None:
        super().__init__()
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pixel_shuffle(x, self.r)


def _det_input(shape):
    g = torch.Generator().manual_seed(1)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(r, x: torch.Tensor):
    ep = torch.export.export(PixelShuffleModule(r).eval(), (x,))
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


class PixelShuffleTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, r) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(r, _det_input(shape))
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (pixel_shuffle {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "pixel_shuffle"),
                    f"pixel_shuffle not delegated (fell back to CPU) for {name}",
                )
