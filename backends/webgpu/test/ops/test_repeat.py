# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.repeat.default` module + configs for the WebGPU op-test framework.

`RepeatModule` tiles the input along each dim. repeat is pure data movement
(bit-identical), so the suite uses the float32 oracle. `RepeatTest` is the
export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, repeats)
CONFIGS = {
    "tile_1d": ((3,), (2,)),
    "tile_2d": ((2, 3), (2, 2)),
    "prepend_3d": ((2, 3), (1, 3, 2)),
    "prepend_ext": ((2, 3), (2, 3, 1)),
    "tile_3d": ((2, 3, 4), (2, 1, 2)),
}


class RepeatModule(torch.nn.Module):
    def __init__(self, repeats) -> None:
        super().__init__()
        self.repeats = repeats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(*self.repeats)


def _det_input(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(repeats, x: torch.Tensor):
    ep = torch.export.export(RepeatModule(repeats).eval(), (x,))
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


class RepeatTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, repeats) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(repeats, _det_input(shape))
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (repeat {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "repeat"),
                    f"repeat not delegated (fell back to CPU) for {name}",
                )
