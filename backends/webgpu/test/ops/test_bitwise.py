# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.bitwise_and.Tensor` / `aten.bitwise_not.default` (bool) modules + configs.

Both are partitioner-tagged for BOOL inputs only, so the modules derive their
bool operands on-GPU from float inputs (`a > 0` via the delegated `gt.Tensor`
against a baked zero buffer) — the only runtime inputs are float tensors (the
op-test framework is float-input-only). `bitwise_and` on bool is identical to
`logical_and` (shares the handler); `bitwise_not` is the bool NOT (1-x). Output
is bool (byte-exact golden). `BitwiseTest` is the export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class BitwiseAndModule(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.register_buffer("z", torch.zeros(shape))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_and(a > self.z, b > self.z)


class BitwiseNotModule(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.register_buffer("z", torch.zeros(shape))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(a > self.z)


def _bw_gen(seed):
    # Distinct per-input seed so derived bool masks differ (bitwise_and).
    def g(shape):
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    return g


bw_gen_a = _bw_gen(0)
bw_gen_b = _bw_gen(1)


# All shapes have numel % 4 == 0 (bool tensors pack 4 bytes/word).
SHAPES = [(4, 8), (2, 3, 8), (16, 16)]


class BitwiseTest(unittest.TestCase):
    def _assert_delegates(self, mod, inputs, op_name, shape) -> None:
        ep = torch.export.export(mod.eval(), inputs)
        edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
        et = edge.to_executorch()
        deleg = any(
            d.id == "VulkanBackend"
            for plan in et.executorch_program.execution_plan
            for d in plan.delegates
        )
        self.assertTrue(deleg, f"Expected VulkanBackend delegate ({op_name} {shape})")
        gm = edge.exported_program().graph_module
        self.assertTrue(
            all(op_name not in str(getattr(n, "target", "")) for n in gm.graph.nodes),
            f"{op_name} fell back to CPU for {shape}",
        )

    def test_export_delegates(self) -> None:
        for shape in SHAPES:
            with self.subTest(shape=shape):
                a = bw_gen_a(shape)
                b = bw_gen_b(shape)
                self._assert_delegates(
                    BitwiseAndModule(shape), (a, b), "bitwise_and", shape
                )
                self._assert_delegates(
                    BitwiseNotModule(shape), (a,), "bitwise_not", shape
                )
