# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.logical_or.default` / `aten.bitwise_or.Tensor` (bool) modules + configs.

Mirrors the logical_and/bitwise_and tests: the modules derive their two bool
operands on-GPU from float inputs (`a > 0`, `b > 0` via the delegated `gt.Tensor`
against a baked zero buffer), so the only runtime inputs are the two float
tensors (the op-test framework is float-input-only). `a`/`b` use distinct seeds
so the two bool masks differ (each ~50% True, independent -> OR ~75% True), a
real mix that a wrong op (e.g. AND) would fail. `bitwise_or` on bool is identical
to `logical_or` (shares the handler). Output is bool (byte-exact golden).
`LogicalOrTest` is the export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class LogicalOrModule(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.register_buffer("z", torch.zeros(shape))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logical_or(a > self.z, b > self.z)


class BitwiseOrModule(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.register_buffer("z", torch.zeros(shape))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_or(a > self.z, b > self.z)


def _lo_gen(seed):
    # Distinct per-input seed so the two derived bool masks differ.
    def g(shape):
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    return g


lo_gen_a = _lo_gen(0)
lo_gen_b = _lo_gen(1)


# All shapes have numel % 4 == 0 (bool tensors pack 4 bytes/word).
SHAPES = [(4, 8), (2, 3, 8), (16, 16)]


class LogicalOrTest(unittest.TestCase):
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
                a = lo_gen_a(shape)
                b = lo_gen_b(shape)
                self._assert_delegates(
                    LogicalOrModule(shape), (a, b), "logical_or", shape
                )
                self._assert_delegates(
                    BitwiseOrModule(shape), (a, b), "bitwise_or", shape
                )
