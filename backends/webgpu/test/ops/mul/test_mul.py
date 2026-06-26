# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class MulModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class MulSelfModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class MulChainedModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = x * y
        z = z * x
        z = z * y
        return z


class TestMul(unittest.TestCase):
    """fp32 torch.mul export tests via VulkanPartitioner."""

    def _export_and_check(self, model, example_inputs) -> None:
        ep = torch.export.export(model, example_inputs)
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()

        found_vulkan = False
        for plan in et_program.executorch_program.execution_plan:
            for delegate in plan.delegates:
                if delegate.id == "VulkanBackend":
                    found_vulkan = True
                    break
        self.assertTrue(found_vulkan, "Expected VulkanBackend delegate in .pte")
        self.assertGreater(len(et_program.buffer), 100)

    def test_mul_2d(self) -> None:
        self._export_and_check(MulModule(), (torch.randn(4, 4), torch.randn(4, 4)))

    def test_mul_3d(self) -> None:
        self._export_and_check(
            MulModule(), (torch.randn(2, 3, 4), torch.randn(2, 3, 4))
        )

    def test_mul_4d(self) -> None:
        self._export_and_check(
            MulModule(), (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        )

    def test_mul_broadcast_last_dim(self) -> None:
        self._export_and_check(MulModule(), (torch.randn(4, 4), torch.randn(4, 1)))

    def test_mul_broadcast_first_dim(self) -> None:
        self._export_and_check(MulModule(), (torch.randn(4, 4), torch.randn(1, 4)))

    def test_mul_self(self) -> None:
        self._export_and_check(MulSelfModule(), (torch.randn(4, 4),))

    def test_mul_chained(self) -> None:
        self._export_and_check(
            MulChainedModule(), (torch.randn(4, 4), torch.randn(4, 4))
        )


def _mul_range(shape) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-3.0, 3.0, n, dtype=torch.float32).reshape(shape)


def _mul_offset_range(shape) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(0.25, 2.25, n, dtype=torch.float32).reshape(shape)


if __name__ == "__main__":
    unittest.main()
