# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class ArangeFloatModule(torch.nn.Module):
    """fp32 arange folded into the graph via an add against the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(0.0, 16.0, 1.0)


class ArangeIntModule(torch.nn.Module):
    """Integer arange — the shape the ViT positional-encoding path produces."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(0, 16, 1).to(torch.float32)


class ArangeStepModule(torch.nn.Module):
    """Non-unit step, to cover the start + i * step arithmetic."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.arange(3.0, 35.0, 2.0)


class TestArange(unittest.TestCase):
    """aten.arange.start_step export tests — uses VulkanPartitioner since the
    WebGPU runtime directly consumes the Vulkan delegate (VK00 FlatBuffer)."""

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

    def test_arange_float(self) -> None:
        self._export_and_check(ArangeFloatModule(), (torch.randn(16),))

    def test_arange_int(self) -> None:
        self._export_and_check(ArangeIntModule(), (torch.randn(16),))

    def test_arange_step(self) -> None:
        self._export_and_check(ArangeStepModule(), (torch.randn(16),))


if __name__ == "__main__":
    unittest.main()
