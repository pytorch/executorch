# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from typing import Tuple

import torch
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir.backend.partitioner import PartitionResult
from executorch.exir.delegate import executorch_call_delegate
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export import export
from torch.fx.passes.utils.fuser_utils import validate_partition


class TestCudaPartitioner(unittest.TestCase):
    """
    Test CUDA partitioner functionality.

    A fully delegatable graph collapses to a single partition. When a
    non-delegated node splits the delegatable ops, the partitioner emits one
    convex partition per island.
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

        # Verify all tagged constants share the (single) partition's tag.
        self.assertEqual(len(partition_result.partition_tags), 1)
        expected_tag = next(iter(partition_result.partition_tags))
        for node in constant_placeholders:
            actual_tag = node.meta.get("delegation_tag")
            self.assertEqual(
                actual_tag,
                expected_tag,
                f"Constant placeholder {node.name} has tag '{actual_tag}' but expected '{expected_tag}'",
            )

    def test_does_not_retag_already_lowered_delegate(self) -> None:
        """
        A node already lowered by a previous partitioner appears as an
        executorch_call_delegate call plus its output getitem. The CUDA
        partitioner must not re-tag those, so it can run after another backend
        (e.g. TensorRT) and only claim the remaining ops.
        """

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        exported_program = export(AddModule(), (torch.randn(3, 4),), strict=True)
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        placeholder = next(n for n in graph.nodes if n.op == "placeholder")
        aten_node = next(
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target != operator.getitem
        )

        # Splice in a fake, already-lowered delegate (call + output getitem), as a
        # preceding partitioner (e.g. TensorRT) would have produced.
        graph_module.lowered_module_0 = torch.nn.Module()
        with graph.inserting_before(aten_node):
            lowered = graph.get_attr("lowered_module_0")
            delegate = graph.call_function(
                executorch_call_delegate, (lowered, placeholder)
            )
            delegate_output = graph.call_function(operator.getitem, (delegate, 0))
        graph.lint()

        CudaPartitioner([]).partition(exported_program)

        self.assertNotIn("delegation_tag", delegate.meta)
        self.assertNotIn("delegation_tag", delegate_output.meta)
        self.assertIn("delegation_tag", aten_node.meta)

    def test_does_not_tag_constant_used_only_by_prior_delegate(self) -> None:
        """
        A constant whose only consumer is a previously lowered delegate must stay
        untagged. Tagging it would give it this partition's tag while its user
        keeps the prior delegate's, which backend lowering rejects. Only ops this
        partitioner claims and genuinely unused constants may be tagged.
        """

        class AddModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("w", torch.randn(3, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.w

        exported_program = export(AddModule(), (torch.randn(3, 4),), strict=True)
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        buffer_placeholder = next(
            n
            for n in graph.nodes
            if n.op == "placeholder" and is_buffer(exported_program, n)
        )
        input_placeholder = next(
            n
            for n in graph.nodes
            if n.op == "placeholder" and not is_buffer(exported_program, n)
        )
        aten_node = next(
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target != operator.getitem
        )

        # Make the buffer feed only a fake, already-lowered delegate (as a
        # preceding TensorRT partition would): rewire the aten op off the buffer,
        # then splice the delegate consuming it.
        aten_node.replace_input_with(buffer_placeholder, input_placeholder)
        graph_module.lowered_module_0 = torch.nn.Module()
        with graph.inserting_before(aten_node):
            lowered = graph.get_attr("lowered_module_0")
            delegate = graph.call_function(
                executorch_call_delegate, (lowered, buffer_placeholder)
            )
            graph.call_function(operator.getitem, (delegate, 0))
        graph.lint()

        CudaPartitioner([]).partition(exported_program)

        self.assertNotIn("delegation_tag", buffer_placeholder.meta)
        self.assertNotIn("delegation_tag", delegate.meta)
        self.assertIn("delegation_tag", aten_node.meta)

    def test_multiple_partitions_for_split_graph(self) -> None:
        """Ops split by a non-delegated node must land in separate partitions.

        One tag over the disconnected islands would be non-convex and fail fusion.
        """

        class TwoAddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = x + 1.0
                return a + 2.0

        exported_program = export(TwoAddModule(), (torch.randn(3, 4),), strict=True)
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        add_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target != operator.getitem
        ]
        first_add, second_add = add_nodes[0], add_nodes[1]

        # Splice an already-lowered region between the two adds so the second add
        # depends on the first only through that non-delegated node.
        graph_module.lowered_module_0 = torch.nn.Module()
        with graph.inserting_before(second_add):
            lowered = graph.get_attr("lowered_module_0")
            delegate = graph.call_function(
                executorch_call_delegate, (lowered, first_add)
            )
            delegate_output = graph.call_function(operator.getitem, (delegate, 0))
        second_add.replace_input_with(first_add, delegate_output)
        graph.lint()

        result = CudaPartitioner([]).partition(exported_program)

        # Separated by the delegate, the adds must land in different partitions.
        self.assertEqual(len(result.partition_tags), 2)
        self.assertIn("delegation_tag", first_add.meta)
        self.assertIn("delegation_tag", second_add.meta)
        self.assertNotEqual(
            first_add.meta["delegation_tag"], second_add.meta["delegation_tag"]
        )
        self.assertNotIn("delegation_tag", delegate.meta)
        self.assertNotIn("delegation_tag", delegate_output.meta)

        # Each partition must be convex on its own so fusion does not cycle.
        for tag in result.partition_tags:
            tagged = [
                n
                for n in exported_program.graph.nodes
                if n.meta.get("delegation_tag") == tag
            ]
            self.assertTrue(validate_partition(tagged))

    def test_control_flow_get_attr_shares_op_tag(self) -> None:
        """A control-flow op's branch get_attrs must share the op's partition tag.

        They are not call_function nodes, so the capability partitioner does not
        claim them; they must be lowered into the same submodule as the op.
        """

        class CondModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.cond(x.sum() > 0, torch.sin, torch.cos, (x,))

        exported_program = export(CondModule(), (torch.randn(3, 4),), strict=True)
        result = CudaPartitioner([]).partition(exported_program)

        cond_node = next(
            n
            for n in exported_program.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.higher_order.cond
        )
        branch_get_attrs = [
            arg
            for arg in cond_node.args
            if isinstance(arg, torch.fx.Node) and arg.op == "get_attr"
        ]

        self.assertEqual(len(branch_get_attrs), 2)
        self.assertIn(cond_node.meta["delegation_tag"], result.partition_tags)
        for get_attr in branch_get_attrs:
            self.assertEqual(
                get_attr.meta.get("delegation_tag"),
                cond_node.meta["delegation_tag"],
            )

    def test_shared_constant_across_partitions(self) -> None:
        """A constant read by two partitions is claimed, not dropped.

        tag_constant_data assigns it one partition's tag; backend lowering later
        duplicates it per consumer, so partitioning must not crash or drop it.
        """

        class SharedWeightModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("w", torch.randn(3, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x + self.w) + self.w

        exported_program = export(
            SharedWeightModule(), (torch.randn(3, 4),), strict=True
        )
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        add_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target != operator.getitem
        ]
        first_add, second_add = add_nodes[0], add_nodes[1]

        # Split the two adds (both reading w) with an already-lowered region.
        graph_module.lowered_module_0 = torch.nn.Module()
        with graph.inserting_before(second_add):
            lowered = graph.get_attr("lowered_module_0")
            delegate = graph.call_function(
                executorch_call_delegate, (lowered, first_add)
            )
            delegate_output = graph.call_function(operator.getitem, (delegate, 0))
        second_add.replace_input_with(first_add, delegate_output)
        graph.lint()

        result = CudaPartitioner([]).partition(exported_program)

        # Two islands, and the shared buffer is claimed by one of them, not dropped.
        self.assertEqual(len(result.partition_tags), 2)
        buffer_placeholder = next(
            n
            for n in graph.nodes
            if n.op == "placeholder" and is_buffer(exported_program, n)
        )
        self.assertIn(
            buffer_placeholder.meta.get("delegation_tag"), result.partition_tags
        )
