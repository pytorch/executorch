# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.native_group_norm.default` module + configs for the op-test framework.

`GroupNormModule` returns the full (out, mean, rstd) tuple via
`torch.native_group_norm`, so the multi-output golden path verifies BOTH the
reduce pass (mean/rstd) and the normalize pass (out). `GroupNormTest` is the
export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> num_channels, num_groups, input_shape
CONFIGS = {
    "c4_g2": (4, 2, (2, 4, 3, 5)),
    "c6_g3": (6, 3, (1, 6, 2, 2)),
    "c4_g1": (4, 1, (1, 4, 3, 3)),
}


class GroupNormModule(torch.nn.Module):
    def __init__(self, num_channels, num_groups) -> None:
        super().__init__()
        self.num_groups = num_groups
        g = torch.Generator().manual_seed(0)
        self.register_buffer("w", torch.randn(num_channels, generator=g))
        self.register_buffer("b", torch.randn(num_channels, generator=g))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        return torch.native_group_norm(
            x, self.w, self.b, n, c, h * w, self.num_groups, self.eps
        )


def _det_input(shape):
    g = torch.Generator().manual_seed(1)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(num_channels, num_groups, x: torch.Tensor):
    ep = torch.export.export(GroupNormModule(num_channels, num_groups).eval(), (x,))
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


class GroupNormTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (num_channels, num_groups, shape) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(num_channels, num_groups, _det_input(shape))
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (group_norm {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "native_group_norm"),
                    f"group_norm not delegated (fell back to CPU) for {name}",
                )
