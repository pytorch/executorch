# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir.backend.compile_spec_schema import CompileSpec
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
        compile_specs = [CompileSpec("cuda_compile_options", b"")]
        partitioner = CudaPartitioner(compile_specs)

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
