# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.logical_and.default` module + configs for the WebGPU op-test framework.

`LogicalAndModule` derives its two bool operands on-GPU from float inputs
(`a > 0`, `b > 0` via the delegated `gt.Tensor` against a baked zero buffer), so
the only runtime inputs are the two float tensors (the op-test framework is
float-input-only). `a`/`b` use distinct seeds so the two bool masks differ (each
~50% True, independent -> AND ~25% True), a real mix that a wrong op (e.g. OR)
would fail. Output is bool (byte-exact golden). `LogicalAndTest` is the
export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class LogicalAndModule(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.register_buffer("z", torch.zeros(shape))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(a > self.z, b > self.z)


def _la_gen(seed):
    # Distinct per-input seed so the two derived bool masks differ.
    def g(shape):
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    return g


la_gen_a = _la_gen(0)
la_gen_b = _la_gen(1)


# All shapes have numel % 4 == 0 (bool tensors pack 4 bytes/word).
SHAPES = [(4, 8), (2, 3, 8), (16, 16)]


class LogicalAndTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for shape in SHAPES:
            with self.subTest(shape=shape):
                a = la_gen_a(shape)
                b = la_gen_b(shape)
                ep = torch.export.export(LogicalAndModule(shape).eval(), (a, b))
                edge = to_edge_transform_and_lower(
                    ep, partitioner=[VulkanPartitioner()]
                )
                et = edge.to_executorch()
                deleg = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(deleg, f"Expected VulkanBackend delegate ({shape})")
                gm = edge.exported_program().graph_module
                self.assertTrue(
                    all(
                        "logical_and" not in str(getattr(n, "target", ""))
                        for n in gm.graph.nodes
                    ),
                    f"logical_and fell back to CPU for {shape}",
                )
