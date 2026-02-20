# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.apple.metal.metal_backend import MetalBackend
from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
from executorch.exir.backend.partitioner import PartitionResult
from torch.export import export


class TestMetalPartitioner(unittest.TestCase):
    """
    Test Metal partitioner functionality.

    After Metal partitioning, there should be exactly one partitioned graph that contains
    all operators from the input graph. This means all operators should be tagged with
    the same delegation tag, indicating they will all be executed by the Metal backend.
    """

    def _get_partition_result(
        self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]
    ) -> PartitionResult:
        """Helper method to get partition result for a given module."""
        # Export the model
        exported_program = export(module, inputs, strict=True)

        # Create partitioner with compile specs
        compile_specs = [MetalBackend.generate_method_name_compile_spec("forward")]
        partitioner = MetalPartitioner(compile_specs)

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
        Test that Metal partitioner creates exactly one partition containing all operators.
        Simple element-wise addition should result in a single graph with all ops tagged identically.
        """

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        # Create test inputs
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Get partition result
        partition_result = self._get_partition_result(AddModule(), (x, y))

        # Verify it's fully partitioned
        self.assertTrue(
            self._check_fully_partitioned(partition_result),
            "Expected all operations to be in a single partition",
        )

        # Verify exactly one partition tag exists
        self.assertEqual(
            len(partition_result.partition_tags),
            1,
            "Expected exactly one partition tag for fully delegated graph",
        )

    def test_linear_partition(self):
        """
        Test Metal partitioner with a linear layer.
        All matrix operations should be in a single partition.
        """

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        # Create test input
        x = torch.randn(2, 10)

        # Get partition result
        partition_result = self._get_partition_result(LinearModule(), (x,))

        # Verify it's fully partitioned
        self.assertTrue(
            self._check_fully_partitioned(partition_result),
            "Expected all operations to be in a single partition",
        )

    def test_ops_to_not_decompose(self):
        """
        Test that ops_to_not_decompose returns all call_function ops.
        Metal backend should handle decomposition via AOTInductor.
        """

        class SimpleModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.relu(x + 1.0)

        # Create test input
        x = torch.randn(2, 3)

        # Export the model
        exported_program = export(SimpleModule(), (x,), strict=True)

        # Create partitioner
        compile_specs = [MetalBackend.generate_method_name_compile_spec("forward")]
        partitioner = MetalPartitioner(compile_specs)

        # Get ops to not decompose
        ops_to_not_decompose, _ = partitioner.ops_to_not_decompose(exported_program)

        # Verify it returns a list
        self.assertIsInstance(ops_to_not_decompose, list)

        # All call_function ops should be in the list
        call_function_ops = [
            node.target
            for node in exported_program.graph.nodes
            if node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
        ]

        self.assertEqual(
            set(ops_to_not_decompose),
            set(call_function_ops),
            "ops_to_not_decompose should contain all call_function ops",
        )


if __name__ == "__main__":
    unittest.main()
