# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delegation coverage for logical AND with packed truth-table inputs."""

import math
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


LOGICAL_BINARY_CASES = (
    ("2d", (4, 8)),
    ("3d", (2, 3, 8)),
    ("sq", (16, 16)),
    ("words63", (252,)),
    ("words64", (256,)),
    ("words65", (260,)),
)


def _logical_binary_gen(pattern):
    def generate(shape):
        numel = math.prod(shape)
        if numel == 0 or numel % len(pattern) != 0:
            raise ValueError("logical-binary test shapes must have numel % 4 == 0")
        return (
            torch.tensor(pattern, dtype=torch.float32)
            .repeat(numel // len(pattern))
            .reshape(shape)
        )

    return generate


logical_binary_gen_a = _logical_binary_gen((-1.0, -1.0, 1.0, 1.0))
logical_binary_gen_b = _logical_binary_gen((-1.0, 1.0, -1.0, 1.0))


class LogicalAndTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for case_name, shape in LOGICAL_BINARY_CASES:
            with self.subTest(case=case_name, shape=shape):
                a = logical_binary_gen_a(shape)
                b = logical_binary_gen_b(shape)
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
