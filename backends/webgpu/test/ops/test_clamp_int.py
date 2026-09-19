# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class ClampIntModule(torch.nn.Module):
    """Integer clamp with both bounds, as index arithmetic produces."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1)
        return x + torch.clamp(idx, 2, 11).to(torch.float32)


class ClampIntMinOnlyModule(torch.nn.Module):
    """Only a lower bound; the upper becomes +inf and must saturate to INT32_MAX."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1)
        return x + torch.clamp(idx, min=5).to(torch.float32)


class ClampIntMaxOnlyModule(torch.nn.Module):
    """Only an upper bound; the lower becomes -inf and must saturate to INT32_MIN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1)
        return x + torch.clamp(idx, max=9).to(torch.float32)


class TestClampInt(unittest.TestCase):
    """Integer aten.clamp.default export tests. fp32 clamp is covered by the
    declarative unary suite; these pin the int32 shader variant, which the ViT
    positional-encoding path needs."""

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

    def test_clamp_int_both_bounds(self) -> None:
        self._export_and_check(ClampIntModule(), (torch.randn(16),))

    def test_clamp_int_min_only(self) -> None:
        self._export_and_check(ClampIntMinOnlyModule(), (torch.randn(16),))

    def test_clamp_int_max_only(self) -> None:
        self._export_and_check(ClampIntMaxOnlyModule(), (torch.randn(16),))


if __name__ == "__main__":
    unittest.main()
