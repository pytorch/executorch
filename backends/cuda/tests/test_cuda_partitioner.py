# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir.backend.partitioner import PartitionResult
from torch.export import export


class TestCudaPartitioner(unittest.TestCase):
    """
    Test CUDA partitioner functionality.

    After CUDA partitioning, there should be exactly one partitioned graph that contains
    all operators from the input graph. This means all operators should be tagged with
    the same delegation tag, indicating they will all be executed by the CUDA backend.
    """

    def _get_partition_result(
        self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]
    ) -> PartitionResult:
        """Helper method to get partition result for a given module."""
        # Export the model
        exported_program = export(module, inputs, strict=True)

        # Create partitioner and compile specs
        partitioner = CudaPartitioner([])

        # Get partition result
        partition_result = partitioner.partition(exported_program)

        # Verify partition result structure
        self.assertIsNotNone(partition_result)
        self.assertTrue(hasattr(partition_result, "tagged_exported_program"))
        self.assertTrue(hasattr(partition_result, "partition_tags"))

        return partition_result

    def _check_fully_partitioned(self, partition_result: PartitionResult) -> bool:
        """Check if the graph is fully partitioned (all operators have the same tag)."""
        tagged_nodes = []
        untagged_ops = []

        for node in partition_result.tagged_exported_program.graph.nodes:
            if node.op == "call_function":
                if hasattr(node, "meta") and "delegation_tag" in node.meta:
                    tagged_nodes.append(node)
                else:
                    untagged_ops.append(node)

        # Check if we have any tagged nodes
        if not tagged_nodes:
            return False

        # Check if all tagged nodes have the same tag
        first_tag = tagged_nodes[0].meta["delegation_tag"]
        all_same_tag = all(
            node.meta.get("delegation_tag") == first_tag for node in tagged_nodes
        )

        # Should have no untagged operations for full partitioning
        fully_partitioned = len(untagged_ops) == 0 and all_same_tag

        return fully_partitioned

    def test_simple_add_partition(self):
        """
        Test that CUDA partitioner creates exactly one partition containing all operators.
        Simple element-wise addition should result in a single graph with all ops tagged identically.
        """

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        module = AddModule()
        inputs = (torch.randn(3, 4), torch.randn(3, 4))

        partition_result = self._get_partition_result(module, inputs)
        fully_partitioned = self._check_fully_partitioned(partition_result)

        self.assertTrue(
            fully_partitioned,
            "Graph should be fully partitioned with all operators having the same tag",
        )

    def test_conv2d_partition(self):
        """
        Test that CUDA partitioner creates exactly one partition containing all operators.
        Conv2D operation should result in a single graph with all ops tagged identically.
        """

        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        module = Conv2dModule()
        inputs = (torch.randn(1, 3, 32, 32),)

        partition_result = self._get_partition_result(module, inputs)
        fully_partitioned = self._check_fully_partitioned(partition_result)

        self.assertTrue(
            fully_partitioned,
            "Graph should be fully partitioned with all operators having the same tag",
        )

    def test_linear_partition(self):
        """
        Test that CUDA partitioner creates exactly one partition containing all operators.
        Linear layer operation should result in a single graph with all ops tagged identically.
        """

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        module = LinearModule()
        inputs = (torch.randn(8, 128),)

        partition_result = self._get_partition_result(module, inputs)
        fully_partitioned = self._check_fully_partitioned(partition_result)

        self.assertTrue(
            fully_partitioned,
            "Graph should be fully partitioned with all operators having the same tag",
        )

    def test_unused_constant_tagging(self):
        """
        Test that constant nodes without users are properly tagged with delegation_tag.

        When a graph contains constants (parameters, buffers, or lifted tensor constants)
        that are not used by any operations, the CUDA partitioner should still tag them
        with the delegation_tag. This ensures all constant data is properly handled during
        delegation, even if they have no users in the graph.
        """

        class ModuleWithUnusedConst(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Register a buffer that won't be used in forward
                self.register_buffer("unused_buffer", torch.randn(10, 10))
                # Also register a used parameter
                self.weight = torch.nn.Parameter(torch.randn(5, 5))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Only use the weight parameter, not the unused_buffer
                return x + self.weight

        module = ModuleWithUnusedConst()
        inputs = (torch.randn(5, 5),)

        # Get partition result
        partition_result = self._get_partition_result(module, inputs)

        # Find all placeholder nodes (these represent constants, parameters, buffers, and inputs)
        constant_placeholders = []
        input_placeholders = []

        for node in partition_result.tagged_exported_program.graph.nodes:
            if node.op == "placeholder":
                # Check if this is a constant (param, buffer, or lifted tensor constant)
                from torch._export.utils import (
                    is_buffer,
                    is_lifted_tensor_constant,
                    is_param,
                )

                is_constant = (
                    is_param(partition_result.tagged_exported_program, node)
                    or is_buffer(partition_result.tagged_exported_program, node)
                    or is_lifted_tensor_constant(
                        partition_result.tagged_exported_program, node
                    )
                )

                if is_constant:
                    constant_placeholders.append(node)
                else:
                    input_placeholders.append(node)

        # Verify we have constant placeholders
        self.assertGreater(
            len(constant_placeholders),
            0,
            "Expected to find constant placeholders in the graph",
        )

        # Check that all constant placeholders are tagged, including unused ones
        untagged_constants = []
        for node in constant_placeholders:
            if "delegation_tag" not in node.meta:
                untagged_constants.append(node.name)

        self.assertEqual(
            len(untagged_constants),
            0,
            f"All constant placeholders should be tagged. Found untagged constants: {untagged_constants}",
        )

        # Verify all tagged constants have the expected tag
        expected_tag = "tag0"
        for node in constant_placeholders:
            actual_tag = node.meta.get("delegation_tag")
            self.assertEqual(
                actual_tag,
                expected_tag,
                f"Constant placeholder {node.name} has tag '{actual_tag}' but expected '{expected_tag}'",
            )
