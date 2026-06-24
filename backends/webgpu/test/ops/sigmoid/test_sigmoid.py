# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class SigmoidModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class SigmoidChainedModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.sigmoid(x))


class TestSigmoid(unittest.TestCase):
    """fp32 torch.sigmoid export tests via VulkanPartitioner."""

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

    def test_sigmoid_1d(self) -> None:
        self._export_and_check(SigmoidModule(), (torch.randn(17),))

    def test_sigmoid_2d(self) -> None:
        self._export_and_check(SigmoidModule(), (torch.randn(4, 4),))

    def test_sigmoid_4d(self) -> None:
        self._export_and_check(SigmoidModule(), (torch.randn(1, 2, 3, 4),))

    def test_sigmoid_chained(self) -> None:
        self._export_and_check(SigmoidChainedModule(), (torch.randn(4, 4),))


def _sigmoid_range(shape) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-8.0, 8.0, n, dtype=torch.float32).reshape(shape)


def _sigmoid_wide_range(shape) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-20.0, 20.0, n, dtype=torch.float32).reshape(shape)


if __name__ == "__main__":
    unittest.main()
