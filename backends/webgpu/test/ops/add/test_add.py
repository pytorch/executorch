# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class AddModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class AddSelfModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class AddScalarModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 3.0


class AddChainedModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = x + y
        z = z + x
        z = z + y
        return z


class TestAdd(unittest.TestCase):
    """fp32 torch.add export tests — uses VulkanPartitioner since the WebGPU
    runtime directly consumes the Vulkan delegate (VK00 FlatBuffer)."""

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

    def test_add_2d(self) -> None:
        self._export_and_check(AddModule(), (torch.randn(4, 4), torch.randn(4, 4)))

    def test_add_3d(self) -> None:
        self._export_and_check(
            AddModule(), (torch.randn(2, 3, 4), torch.randn(2, 3, 4))
        )

    def test_add_4d(self) -> None:
        self._export_and_check(
            AddModule(), (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        )

    def test_add_broadcast_last_dim(self) -> None:
        self._export_and_check(AddModule(), (torch.randn(4, 4), torch.randn(4, 1)))

    def test_add_broadcast_first_dim(self) -> None:
        self._export_and_check(AddModule(), (torch.randn(4, 4), torch.randn(1, 4)))

    def test_add_self(self) -> None:
        self._export_and_check(AddSelfModule(), (torch.randn(4, 4),))

    def test_add_scalar(self) -> None:
        self._export_and_check(AddScalarModule(), (torch.randn(4, 4),))

    def test_add_chained(self) -> None:
        self._export_and_check(
            AddChainedModule(), (torch.randn(4, 4), torch.randn(4, 4))
        )


def export_add_model(output_path: str) -> None:
    """Export a simple add model to .pte for native runtime testing."""
    model = AddModule()
    example_inputs = (torch.randn(1024, 1024), torch.randn(1024, 1024))
    ep = torch.export.export(model, example_inputs)
    et_program = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"Exported {output_path}")


if __name__ == "__main__":
    unittest.main()
