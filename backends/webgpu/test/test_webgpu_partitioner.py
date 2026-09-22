# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.webgpu.partitioner import WebGPUPartitioner
from executorch.backends.webgpu.test.tester import Partition, ToEdgeTransformAndLower
from executorch.exir import (
    ExecutorchProgramManager,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


class AddModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class TestWebGPUPartitioner(unittest.TestCase):
    def _export_add(self) -> ExportedProgram:
        return torch.export.export(
            AddModule(),
            (torch.ones(2, 3), torch.full((2, 3), 2.0)),
        )

    def _partition(self, partitioner: Partitioner) -> PartitionResult:
        return partitioner.partition(to_edge(self._export_add()).exported_program())

    def _lower(self, partitioner: Partitioner) -> ExecutorchProgramManager:
        return to_edge_transform_and_lower(
            self._export_add(), partitioner=[partitioner]
        ).to_executorch()

    def _delegate_ids(self, program: ExecutorchProgramManager) -> List[str]:
        return [
            delegate.id
            for plan in program.executorch_program.execution_plan
            for delegate in plan.delegates
        ]

    def _assert_same_partitioning(
        self,
        webgpu_partitioner: WebGPUPartitioner,
        vulkan_partitioner: VulkanPartitioner,
    ) -> PartitionResult:
        webgpu_result = self._partition(webgpu_partitioner)
        vulkan_result = self._partition(vulkan_partitioner)
        self.assertEqual(webgpu_result.partition_tags, vulkan_result.partition_tags)
        return webgpu_result

    def test_lowers_identically_with_vulkan_backend_id(self) -> None:
        webgpu_program = self._lower(WebGPUPartitioner())
        vulkan_program = self._lower(VulkanPartitioner())

        self.assertEqual(webgpu_program.buffer, vulkan_program.buffer)
        self.assertEqual(self._delegate_ids(webgpu_program), ["VulkanBackend"])

    def test_forwards_compile_options(self) -> None:
        compile_options = {"skip_bool_tensors": True}
        expected_options = compile_options.copy()

        result = self._assert_same_partitioning(
            WebGPUPartitioner(compile_options=compile_options),
            VulkanPartitioner(compile_options=compile_options),
        )

        self.assertGreater(len(result.partition_tags), 0)
        self.assertEqual(compile_options, expected_options)
        for spec in result.partition_tags.values():
            self.assertEqual(spec.backend_id, "VulkanBackend")
            self.assertIn(
                "skip_bool_tensors", [item.key for item in spec.compile_specs]
            )

    def test_forwards_operator_blocklist(self) -> None:
        operator_blocklist = [exir_ops.edge.aten.add.Tensor]
        expected_blocklist = operator_blocklist.copy()
        self.assertGreater(len(self._partition(WebGPUPartitioner()).partition_tags), 0)

        result = self._assert_same_partitioning(
            WebGPUPartitioner(operator_blocklist=operator_blocklist),
            VulkanPartitioner(operator_blocklist=operator_blocklist),
        )

        self.assertEqual(result.partition_tags, {})
        self.assertEqual(operator_blocklist, expected_blocklist)

    def test_forwards_operator_allowlist(self) -> None:
        operator_allowlist = [exir_ops.edge.aten.mul.Tensor]
        expected_allowlist = operator_allowlist.copy()
        self.assertGreater(len(self._partition(WebGPUPartitioner()).partition_tags), 0)

        result = self._assert_same_partitioning(
            WebGPUPartitioner(operator_allowlist=operator_allowlist),
            VulkanPartitioner(operator_allowlist=operator_allowlist),
        )

        self.assertEqual(result.partition_tags, {})
        self.assertEqual(operator_allowlist, expected_allowlist)

    def test_forwards_nn_module_blocklist(self) -> None:
        nn_module_blocklist = ["AddModule"]
        expected_blocklist = nn_module_blocklist.copy()
        self.assertGreater(len(self._partition(WebGPUPartitioner()).partition_tags), 0)

        result = self._assert_same_partitioning(
            WebGPUPartitioner(nn_module_blocklist=nn_module_blocklist),
            VulkanPartitioner(nn_module_blocklist=nn_module_blocklist),
        )

        self.assertEqual(result.partition_tags, {})
        self.assertEqual(nn_module_blocklist, expected_blocklist)

    def test_forwards_nn_module_allowlist(self) -> None:
        nn_module_allowlist = ["DoesNotMatch"]
        expected_allowlist = nn_module_allowlist.copy()
        self.assertGreater(len(self._partition(WebGPUPartitioner()).partition_tags), 0)

        result = self._assert_same_partitioning(
            WebGPUPartitioner(nn_module_allowlist=nn_module_allowlist),
            VulkanPartitioner(nn_module_allowlist=nn_module_allowlist),
        )

        self.assertEqual(result.partition_tags, {})
        self.assertEqual(nn_module_allowlist, expected_allowlist)

    def test_forwards_ops_to_not_decompose(self) -> None:
        exported_program = self._export_add()
        webgpu_ops, webgpu_filter = WebGPUPartitioner().ops_to_not_decompose(
            exported_program
        )
        vulkan_ops, vulkan_filter = VulkanPartitioner().ops_to_not_decompose(
            exported_program
        )

        self.assertEqual(webgpu_ops, vulkan_ops)
        self.assertIsNotNone(webgpu_filter)
        self.assertIsNotNone(vulkan_filter)
        assert webgpu_filter is not None
        assert vulkan_filter is not None
        node = next(
            node
            for node in exported_program.graph_module.graph.nodes
            if node.op == "call_function"
        )
        self.assertEqual(webgpu_filter(node), vulkan_filter(node))

    def test_webgpu_tester_defaults_to_webgpu_partitioner(self) -> None:
        self.assertIsInstance(Partition().partitioner, WebGPUPartitioner)
        self.assertTrue(
            all(
                isinstance(partitioner, WebGPUPartitioner)
                for partitioner in ToEdgeTransformAndLower().partitioners
            )
        )


if __name__ == "__main__":
    unittest.main()
