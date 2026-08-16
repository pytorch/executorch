# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delegation coverage for logical and bitwise OR truth-table inputs."""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.webgpu.test.ops.test_logical_and import (
    LOGICAL_BINARY_CASES,
    logical_binary_gen_a,
    logical_binary_gen_b,
)
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
        for case_name, shape in LOGICAL_BINARY_CASES:
            with self.subTest(case=case_name, shape=shape):
                a = logical_binary_gen_a(shape)
                b = logical_binary_gen_b(shape)
                self._assert_delegates(
                    LogicalOrModule(shape), (a, b), "logical_or", shape
                )
                self._assert_delegates(
                    BitwiseOrModule(shape), (a, b), "bitwise_or", shape
                )
