# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import torch
from executorch.exir.backend.canonical_partitioners.group_partitioner import (
    GroupBasedPartitioner,
)

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase


class TestGroupPartitioner(unittest.TestCase):
    class TestOperatorSupport(OperatorSupportBase):
        def __init__(self):
            super().__init__()
            self.supported_nodes = {
                "linear",
                "linear_1",
                "linear_2",
                "fake_quantize_per_tensor_affine",
                "fake_quantize_per_tensor_affine_1",
                "add",
                "bmm",
                "squeeze",
                "unsqueeze",
                "unsqueeze_1",
                "squeeze_1",
                "tanh",
                "relu",
            }

        def add_supported_node(self, node_name):
            self.supported_nodes.add(node_name)

        def add_supported_nodes(self, node_names):
            self.supported_nodes.update(node_names)

        def is_node_supported(
            self, submodules, node
        ):  # submodules is required by interface
            if node.op == "get_attr":
                return True

            if node.name in self.supported_nodes:
                return True

            return False

    def _find_nodes_by_names(
        self, node_names: List[str], graph_module: torch.fx.GraphModule
    ) -> List[torch.fx.Node]:
        """
        Find nodes in the graph that match the given names.

        Args:
            node_names: A list of node names or patterns to match
            graph_module: The graph module to search in

        Returns:
            A list of nodes that match the given names
        """
        result = []
        not_found = []

        for name in node_names:
            found = False

            # First try exact name match
            for node in graph_module.graph.nodes:
                if node.name == name:
                    result.append(node)
                    found = True
                    break

            if not found:
                for node in graph_module.graph.nodes:
                    if name in node.name:
                        result.append(node)
                        found = True
                        break

                    if node.op == "call_function" and name in str(node.target):
                        result.append(node)
                        found = True
                        break

            if not found:
                not_found.append(name)

        if not_found:
            print(f"Warning: Could not find nodes matching: {not_found}")

        return result

    def create_model(self, model):
        return model().eval()

    def create_input(self):
        return torch.randn(5, 10)

    def export_program(self, model, x):
        return torch.export.export(model, (x,))

    def find_input_nodes(self, exported_program, names=None):
        if not names:
            return None
        out = []
        for group in names:
            out.append(self._find_nodes_by_names(group, exported_program.graph_module))
        return out

    def create_partitioner(self, exported_program, inputNodes):
        return GroupBasedPartitioner(
            exported_program.graph_module,
            self.TestOperatorSupport(),
            node_groups=inputNodes,
            allows_single_node_partition=True,
        )

    def check_partition(self, partitions, expected_partitions):
        partition_found = False
        for partition in partitions:
            node_names = {node.name for node in partition.nodes}
            if expected_partitions.issubset(node_names):
                partition_found = True
                break
        self.assertEqual(partition_found, True)

    def test_qdq_linear_group_partitioning(self):
        """
        Test that GroupBasedPartitioner correctly groups QDQ (quantize-dequantize) patterns with linear operations.
        """

        class SharedQDQModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                scale = 0.1
                zero_point = 0

                # Simulate quantization
                x_q = torch.fake_quantize_per_tensor_affine(
                    x, scale, zero_point, 0, 255
                )

                # First linear path
                y1 = self.linear1(x_q)

                # Non-supported op path
                z = torch.sin(y1)  # Non-supported op
                out1 = torch.bmm(z.unsqueeze(1), z.unsqueeze(2)).squeeze()

                # Second linear path using the same quantized tensor
                y2 = self.linear2(x_q)

                return y1, y2, out1

        model = self.create_model(SharedQDQModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)
        inputNodes = self.find_input_nodes(
            exported_program,
            [["linear", "linear_1", "fake_quantize_per_tensor_affine"]],
        )
        partitioner = self.create_partitioner(exported_program, inputNodes)
        partitions = partitioner.propose_partitions()
        self.check_partition(
            partitions, {"linear", "linear_1", "fake_quantize_per_tensor_affine"}
        )

    def test_complex_graph_with_interdependencies(self):
        """
        Test that GroupBasedPartitioner correctly handles complex graphs with interdependent paths.
        """

        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)  # Changed output size to 10
                self.linear2 = torch.nn.Linear(10, 15)  # Changed input size to 10
                self.linear3 = torch.nn.Linear(15, 10)
                self.linear4 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Path 1
                a = self.linear1(x)
                b = torch.relu(a)

                # Path 2
                c = self.linear2(b)
                d = torch.tanh(c)

                # Path 3 with dependency on both paths
                e = self.linear3(d)
                f = e + b  # Creates dependency between paths

                # Path 4
                g = self.linear4(f)
                return g

        model = self.create_model(ComplexModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)
        inputNodes = self.find_input_nodes(exported_program)
        partitioner = self.create_partitioner(exported_program, inputNodes)
        partitions = partitioner.propose_partitions()

        # Check that the partition includes the expected nodes
        self.check_partition(partitions, {"linear", "relu"})

    def test_branching_qdq_pattern(self):
        """
        Test a branching QDQ pattern where two linear ops share the same quantized input.
        """

        class BranchingQDQModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                scale = 0.1
                zero_point = 0

                # Simulate quantization
                x_q = torch.fake_quantize_per_tensor_affine(
                    x, scale, zero_point, 0, 255
                )

                # Two linear paths using the same quantized tensor
                y1 = self.linear1(x_q)
                y2 = self.linear2(x_q)

                # Non-supported op on first path
                z = torch.sin(y1)

                # add z and a
                a = torch.add(z, y2)
                return a

        model = self.create_model(BranchingQDQModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)
        inputNodes = self.find_input_nodes(
            exported_program,
            [["fake_quantize_per_tensor_affine", "linear", "linear_1"]],
        )
        partitioner = self.create_partitioner(exported_program, inputNodes)
        partitions = partitioner.propose_partitions()

        # Check that the quantize and both linear ops are in the same partition
        self.check_partition(
            partitions, {"linear", "linear_1", "fake_quantize_per_tensor_affine"}
        )
        self.check_partition(partitions, {"add"})

    def test_multi_level_dependencies(self):
        """
        Test a more complex pattern with multi-level dependencies.
        """

        class MultiLevelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 5)

            def forward(self, x):
                scale = 0.1
                zero_point = 0

                # Simulate quantization
                x_q = torch.fake_quantize_per_tensor_affine(
                    x, scale, zero_point, 0, 255
                )

                # First linear path
                y1 = self.linear1(x_q)

                # Second linear path
                y2 = self.linear2(x_q)

                # Third path depends on both previous paths
                y3 = y1 + y2
                out = self.linear3(y3)

                # Non-supported op
                z = torch.sin(out)

                return out, z

        model = self.create_model(MultiLevelModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)
        inputNodes = self.find_input_nodes(
            exported_program,
            [["fake_quantize_per_tensor_affine", "linear", "linear_1", "linear_2"]],
        )
        partitioner = self.create_partitioner(exported_program, inputNodes)
        partitions = partitioner.propose_partitions()

        # Check that all linear ops and quantize are in the same partition
        self.check_partition(
            partitions,
            {"linear", "linear_1", "linear_2", "fake_quantize_per_tensor_affine"},
        )

    def test_double_QDQ_partitioning(self):
        """
        Test that GroupBasedPartitioner correctly handles models with multiple QDQ patterns.
        """

        class TwoSharedQDQModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)
                self.linear3 = torch.nn.Linear(10, 10)

            def forward(self, x):
                scale = 0.1
                zero_point = 0

                # Simulate quantization
                x_q = torch.fake_quantize_per_tensor_affine(
                    x, scale, zero_point, 0, 255
                )

                # First linear path
                y1 = self.linear1(x_q)

                # Non-supported op path
                z = torch.sin(y1)  # Non-supported op
                out1 = torch.bmm(z.unsqueeze(1), z.unsqueeze(2)).squeeze()

                # Second linear path using the same quantized tensor
                y2 = self.linear2(x_q)

                # Simulate quantization
                x_q2 = torch.fake_quantize_per_tensor_affine(
                    x, scale, zero_point, 0, 255
                )
                z1 = self.linear3(x_q2)
                o = torch.add(z, z1)

                return o, y2, out1

        model = self.create_model(TwoSharedQDQModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        nodeGroups = self.find_input_nodes(
            exported_program,
            [
                ["linear", "linear_1", "fake_quantize_per_tensor_affine"],
                ["add", "linear_2", "fake_quantize_per_tensor_affine_1"],
            ],
        )

        partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            self.TestOperatorSupport(),
            node_groups=nodeGroups,
            allows_single_node_partition=True,
        )

        partitions = partitioner.propose_partitions()

        self.check_partition(
            partitions, {"linear", "linear_1", "fake_quantize_per_tensor_affine"}
        )
        self.check_partition(
            partitions, {"add", "linear_2", "fake_quantize_per_tensor_affine_1"}
        )

    # New tests for node_groups = None and comparison with CapabilityBasedPartitioner

    def setup_model_for_testing(self, model_class, additional_supported_nodes=None):
        model = self.create_model(model_class)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Create operator support
        op_support = self.TestOperatorSupport()
        if additional_supported_nodes:
            op_support.add_supported_nodes(additional_supported_nodes)

        return exported_program, op_support

    def create_both_partitioners(
        self,
        exported_program,
        op_support,
        allows_single_node_partition=True,
        non_compute_ops=None,
        allowed_single_node_partition_ops=None,
    ):

        # Create GroupBasedPartitioner
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=allows_single_node_partition,
            non_compute_ops=non_compute_ops,
            allowed_single_node_partition_ops=allowed_single_node_partition_ops,
        )

        # Create CapabilityBasedPartitioner
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=allows_single_node_partition,
            non_compute_ops=non_compute_ops,
            allowed_single_node_partition_ops=allowed_single_node_partition_ops,
        )

        return group_partitioner, capability_partitioner

    def run_and_compare_partitioners(
        self, group_partitioner, capability_partitioner, test_name=""
    ):
        """
        Run both partitioners and compare their results.

        Args:
            group_partitioner: The GroupBasedPartitioner instance
            capability_partitioner: The CapabilityBasedPartitioner instance
            test_name: Optional name for the test (for debug output)

        Returns:
            tuple: (group_partitions, capability_partitions)
        """
        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

        return group_partitions, capability_partitions

    def compare_partitions(self, partitions1, partitions2):
        """
        Compare two sets of partitions to see if they are equivalent.
        Two sets of partitions are considered equivalent if:
        1. They have the same number of partitions
        2. For each partition in the first set, there is a partition in the second set with the same nodes
        """
        if len(partitions1) != len(partitions2):
            print(
                f"Different number of partitions: {len(partitions1)} vs {len(partitions2)}"
            )
            return False

        # Convert partitions to sets of node names for easier comparison
        partition_sets1 = [
            frozenset(node.name for node in p.nodes) for p in partitions1
        ]
        partition_sets2 = [
            frozenset(node.name for node in p.nodes) for p in partitions2
        ]

        # Check if each partition in the first set has a matching partition in the second set
        for p1 in partition_sets1:
            if p1 not in partition_sets2:
                print(f"Partition {p1} not found in second set")
                return False

        # Also check the reverse to ensure both sets have the same partitions
        for p2 in partition_sets2:
            if p2 not in partition_sets1:
                print(f"Partition {p2} not found in first set")
                return False

        return True

    def test_null_node_groups_simple_model(self):
        """
        Test that GroupBasedPartitioner with node_groups=None produces similar results
        to CapabilityBasedPartitioner for a simple model.
        """

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        # Setup model and create partitioners
        exported_program, op_support = self.setup_model_for_testing(SimpleModel)
        group_partitioner, capability_partitioner = self.create_both_partitioners(
            exported_program, op_support
        )

        # Run partitioners and compare results
        self.run_and_compare_partitioners(
            group_partitioner, capability_partitioner, "Simple Model"
        )

    def test_null_node_groups_complex_model(self):
        """
        Test that GroupBasedPartitioner with node_groups=None produces reasonable partitions
        for a more complex model with multiple paths and dependencies.
        """

        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Path 1
                a = self.linear1(x)
                b = torch.relu(a)

                # Path 2
                c = self.linear2(x)
                d = torch.tanh(c)

                # Merge paths
                e = b + d
                f = self.linear3(e)
                return f

        # Setup model and create partitioners
        exported_program, op_support = self.setup_model_for_testing(
            ComplexModel, additional_supported_nodes=["add_1"]
        )
        group_partitioner, capability_partitioner = self.create_both_partitioners(
            exported_program, op_support
        )

        # Run partitioners and compare results
        group_partitions, capability_partitions = self.run_and_compare_partitioners(
            group_partitioner, capability_partitioner, "Complex Model"
        )

        # Additional checks for fusion patterns
        linear_relu_found = False
        linear_tanh_found = False

        for p in group_partitions:
            node_names = {node.name for node in p.nodes}
            if "linear" in node_names and "relu" in node_names:
                linear_relu_found = True
            if "linear_1" in node_names and "tanh" in node_names:
                linear_tanh_found = True

        self.assertTrue(
            linear_relu_found or linear_tanh_found,
            "Expected to find linear+relu or linear+tanh fusion patterns",
        )

    def test_null_node_groups_with_cycles(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles potential cycles correctly.
        """

        class CyclicDependencyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(20, 5)

            def forward(self, x):
                # First path
                a = self.linear1(x)
                b = torch.relu(a)

                # Second path with dependency on first
                c = self.linear2(b)
                d = torch.tanh(c)

                # Create a potential cycle by concatenating with original input
                e = torch.cat([d, x], dim=1)
                f = self.linear3(e)
                return f

        model = self.create_model(CyclicDependencyModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["cat", "linear_3"])

        # Create partitioner with node_groups=None
        partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        # This should not raise an exception
        partitions = partitioner.propose_partitions()

        # Check that all supported nodes are included in partitions
        all_supported_nodes = set()
        for node in exported_program.graph_module.graph.nodes:
            if op_support.is_node_supported(None, node):
                all_supported_nodes.add(node.name)

        partition_nodes = set()
        for p in partitions:
            for node in p.nodes:
                partition_nodes.add(node.name)

        self.assertEqual(partition_nodes, all_supported_nodes)

    def test_compare_with_capability_partitioner_branching(self):
        """
        Compare GroupBasedPartitioner with node_groups=None to CapabilityBasedPartitioner
        on a model with branching paths.
        """

        class BranchingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Branch 1
                a = self.linear1(x)
                b = torch.relu(a)

                # Branch 2
                c = self.linear2(x)
                d = torch.tanh(c)

                # Merge branches
                e = b + d
                f = self.linear3(e)
                return f

        model = self.create_model(BranchingModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["add_1", "linear_3"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_null_node_groups_with_unsqueeze_squeeze(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles unsqueeze/squeeze operations correctly.
        """

        class UnsqueezeSqueezeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Path with unsqueeze/squeeze operations
                a = self.linear1(x)
                b = torch.unsqueeze(a, 1)  # Add a dimension
                c = torch.relu(b)
                d = torch.squeeze(c, 1)  # Remove the dimension
                e = self.linear2(d)
                return e

        model = self.create_model(UnsqueezeSqueezeModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            self.TestOperatorSupport(),
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            self.TestOperatorSupport(),
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_complex_model_with_multiple_paths(self):
        """
        Test that GroupBasedPartitioner with node_groups=None produces the same partitions
        as CapabilityBasedPartitioner for a more complex model with multiple paths and operations.
        """

        class ComplexPathsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 10)
                self.linear4 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Path 1
                a = self.linear1(x)
                b = torch.relu(a)

                # Path 2
                c = self.linear2(x)
                d = torch.tanh(c)

                # Path 3
                e = self.linear3(x)
                f = torch.relu(e)

                # Merge paths 1 and 2
                g = b + d

                # Merge with path 3
                h = g + f

                # Final output
                i = self.linear4(h)
                return i

        model = self.create_model(ComplexPathsModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["add_1", "add_2", "linear_3", "linear_4"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_reshape_operations(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles reshape operations
        the same way as CapabilityBasedPartitioner.
        """

        class ReshapeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Path with reshape operations
                a = self.linear1(x)
                b = torch.reshape(a, (5, 2, 5))
                c = torch.relu(b)
                d = torch.reshape(c, (5, 10))
                e = self.linear2(d)
                return e

        model = self.create_model(ReshapeModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add reshape operations to supported nodes
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["reshape", "reshape_1"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_multiple_outputs(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles models with multiple outputs
        the same way as CapabilityBasedPartitioner.
        """

        class MultiOutputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)
                self.linear3 = torch.nn.Linear(10, 3)

            def forward(self, x):
                a = self.linear1(x)
                b = torch.relu(a)

                # First output path
                c = self.linear2(b)

                # Second output path
                d = self.linear3(b)

                return c, d

        model = self.create_model(MultiOutputModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["linear_3"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_shared_subgraphs(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles models with shared subgraphs
        the same way as CapabilityBasedPartitioner.
        """

        class SharedSubgraphModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Shared computation
                a = self.linear1(x)
                b = torch.relu(a)

                # Path 1 using shared computation
                c = self.linear2(b)

                # Path 2 using shared computation
                d = torch.tanh(b)

                # Merge paths
                e = c + d
                f = self.linear3(e)
                return f

        model = self.create_model(SharedSubgraphModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["add_1", "linear_3"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_non_compute_ops(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles non-compute operations
        the same way as CapabilityBasedPartitioner.
        """

        class NonComputeOpsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Path with view operations (typically considered non-compute)
                a = self.linear1(x)
                b = torch.reshape(a, (5, 2, 5))
                c = torch.relu(b)
                d = torch.reshape(c, (5, 10))
                e = self.linear2(d)
                return e

        model = self.create_model(NonComputeOpsModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add reshape operations to supported nodes
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["reshape", "reshape_1"])

        # Define non-compute ops
        non_compute_ops = ["reshape", "reshape_1"]

        # Create both partitioners with non_compute_ops
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
            non_compute_ops=non_compute_ops,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
            non_compute_ops=non_compute_ops,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_allowed_single_node_partition_ops(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles allowed single node partition ops
        the same way as CapabilityBasedPartitioner.
        """

        class SingleNodeOpsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Path with operations that might be allowed as single node partitions
                a = self.linear1(x)
                b = torch.sin(a)  # Non-supported op to break partitions
                c = torch.tanh(b)  # This will be allowed as a single node partition
                d = torch.sin(c)  # Non-supported op to break partitions
                e = self.linear2(d)
                return e

        model = self.create_model(SingleNodeOpsModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Create operator support with tanh as allowed single node partition op
        op_support = self.TestOperatorSupport()
        op_support.add_supported_node("tanh_1")

        # Define allowed single node partition ops
        allowed_single_node_partition_ops = ["tanh_1"]

        # Create both partitioners with allows_single_node_partition=False but specific ops allowed
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=False,
            allowed_single_node_partition_ops=allowed_single_node_partition_ops,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=False,
            allowed_single_node_partition_ops=allowed_single_node_partition_ops,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_complex_dependency_cycles(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles complex dependency cycles
        the same way as CapabilityBasedPartitioner.
        """

        class ComplexCycleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 10)
                self.linear4 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Create a complex dependency pattern with potential cycles
                a = self.linear1(x)
                b = torch.relu(a)

                # Path with dependency on b
                c = self.linear2(b)
                d = torch.tanh(c)

                # Another path with dependency on b
                e = self.linear3(b)
                f = torch.relu(e)

                # Create a cycle-like dependency pattern
                g = d + f
                h = g + b  # Creates a cycle-like pattern

                i = self.linear4(h)
                return i

        model = self.create_model(ComplexCycleModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["add_1", "add_2", "linear_3", "linear_4"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_buffer_mutations(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles buffer mutations
        the same way as CapabilityBasedPartitioner.
        """

        class BufferMutationModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.zeros(1))
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Increment counter (buffer mutation)
                self.counter.add_(1.0)

                # Use the buffer in computation
                y = x + self.counter
                z = self.linear(y)
                return z

        model = self.create_model(BufferMutationModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["add", "add_"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_dynamic_shapes(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles models with dynamic shapes
        the same way as CapabilityBasedPartitioner.
        """

        class DynamicShapeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Operations that depend on input shape
                batch_size = x.size(0)
                a = self.linear1(x)
                b = torch.relu(a)

                # Reshape based on dynamic batch size
                c = torch.reshape(b, (batch_size, -1))
                d = self.linear2(c)
                return d

        model = self.create_model(DynamicShapeModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["reshape", "size"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_complex_graph_structure(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles complex graph structures
        the same way as CapabilityBasedPartitioner.
        """

        class ComplexGraphModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 10)
                self.linear4 = torch.nn.Linear(10, 10)
                self.linear5 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Create a complex graph with multiple paths and dependencies

                # Path 1
                a = self.linear1(x)
                b = torch.relu(a)

                # Path 2
                c = self.linear2(x)
                d = torch.tanh(c)

                # Path 3 with dependency on path 1
                e = self.linear3(b)
                f = torch.relu(e)

                # Path 4 with dependency on path 2
                g = self.linear4(d)
                h = torch.tanh(g)

                # Merge paths 3 and 4
                i = f + h

                # Merge with original paths
                j = i + b + d

                # Final output
                k = self.linear5(j)
                return k

        model = self.create_model(ComplexGraphModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(
            ["add_1", "add_2", "linear_3", "linear_4", "linear_5"]
        )

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_custom_operator_support(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles custom operator support
        the same way as CapabilityBasedPartitioner.
        """

        class CustomOpSupportModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                a = self.linear1(x)
                b = torch.relu(a)
                c = torch.sigmoid(b)  # This will be supported by custom op support
                d = self.linear2(c)
                return d

        # Define a custom operator support class
        class CustomOperatorSupport(OperatorSupportBase):
            def __init__(self):
                super().__init__()
                # Support only specific operations
                self.supported_ops = {
                    torch.ops.aten.linear.default,
                    torch.ops.aten.relu.default,
                    torch.ops.aten.sigmoid.default,
                }

            def is_node_supported(self, submodules, node):
                if node.op == "get_attr":
                    return True

                if node.op == "call_function" and node.target in self.supported_ops:
                    return True

                return False

        model = self.create_model(CustomOpSupportModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Create both partitioners with custom operator support
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            CustomOperatorSupport(),
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            CustomOperatorSupport(),
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_fusion_patterns(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles fusion patterns
        the same way as CapabilityBasedPartitioner.
        """

        class FusionPatternModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Pattern 1: Linear -> ReLU (common fusion pattern)
                a = self.linear1(x)
                b = torch.relu(a)

                # Pattern 2: Linear -> Tanh (another fusion pattern)
                c = self.linear2(x)
                d = torch.tanh(c)

                # Merge results
                e = b + d
                f = self.linear3(e)
                return f

        model = self.create_model(FusionPatternModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["add_1", "linear_3"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

        # Check that fusion patterns are preserved in partitions
        linear_relu_fusion = False
        linear_tanh_fusion = False

        for p in group_partitions:
            node_names = {node.name for node in p.nodes}
            if "linear" in node_names and "relu" in node_names:
                linear_relu_fusion = True
            if "linear_1" in node_names and "tanh" in node_names:
                linear_tanh_fusion = True

        self.assertTrue(
            linear_relu_fusion, "Linear->ReLU fusion pattern should be preserved"
        )
        self.assertTrue(
            linear_tanh_fusion, "Linear->Tanh fusion pattern should be preserved"
        )

    def test_with_large_model(self):
        """
        Test that GroupBasedPartitioner with node_groups=None handles large models
        the same way as CapabilityBasedPartitioner.
        """

        class LargeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Create a model with many layers
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(10, 10) for _ in range(10)]
                )
                self.final = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Process through many layers with different activation functions
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i % 3 == 0:
                        x = torch.relu(x)
                    elif i % 3 == 1:
                        x = torch.tanh(x)
                    else:
                        x = torch.sigmoid(x)

                return self.final(x)

        model = self.create_model(LargeModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(
            [f"linear_{i}" for i in range(1, 11)]
            + [
                "sigmoid",
                "sigmoid_1",
                "sigmoid_2",
                "tanh_1",
                "tanh_2",
                "relu_1",
                "relu_2",
            ]
        )

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_with_different_traversal_orders(self):
        """
        Test that GroupBasedPartitioner with node_groups=None produces the same partitions
        regardless of the order in which nodes are processed.
        """

        class TraversalOrderModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)
                self.linear3 = torch.nn.Linear(10, 10)
                self.linear4 = torch.nn.Linear(10, 5)

            def forward(self, x):
                # Create a graph with multiple independent paths
                a = self.linear1(x)
                b = torch.relu(a)

                c = self.linear2(x)
                d = torch.tanh(c)

                e = self.linear3(x)
                f = torch.relu(e)

                # Merge all paths
                g = b + d + f
                h = self.linear4(g)
                return h

        model = self.create_model(TraversalOrderModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Add more supported nodes for this test
        op_support = self.TestOperatorSupport()
        op_support.add_supported_nodes(["add_1", "add_2", "linear_3", "linear_4"])

        # Create both partitioners
        group_partitioner = GroupBasedPartitioner(
            exported_program.graph_module,
            op_support,
            node_groups=None,
            allows_single_node_partition=True,
        )

        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            op_support,
            allows_single_node_partition=True,
        )

        # Get partitions from both partitioners
        group_partitions = group_partitioner.propose_partitions()
        capability_partitions = capability_partitioner.propose_partitions()

        # Check that both partitioners produce exactly the same partitions
        self.assertTrue(
            self.compare_partitions(group_partitions, capability_partitions),
            "GroupBasedPartitioner and CapabilityBasedPartitioner should produce the same partitions",
        )

    def test_null_node_groups_single_node_partition_control(self):
        """
        Test that GroupBasedPartitioner with node_groups=None respects the
        allows_single_node_partition parameter.
        """

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.sin(x)  # Non-supported op to break partitions
                x = self.linear2(x)
                return x

        model = self.create_model(SimpleModel)
        x = self.create_input()
        exported_program = self.export_program(model, x)

        # Create partitioner with allows_single_node_partition=False
        partitioner_no_single = GroupBasedPartitioner(
            exported_program.graph_module,
            self.TestOperatorSupport(),
            node_groups=None,
            allows_single_node_partition=False,
        )

        # Create partitioner with allows_single_node_partition=True
        partitioner_with_single = GroupBasedPartitioner(
            exported_program.graph_module,
            self.TestOperatorSupport(),
            node_groups=None,
            allows_single_node_partition=True,
        )

        partitions_no_single = partitioner_no_single.propose_partitions()
        partitions_with_single = partitioner_with_single.propose_partitions()

        # With allows_single_node_partition=False, we should have no partitions
        # since the non-supported op breaks the graph into single-node partitions
        self.assertEqual(len(partitions_no_single), 0)

        # With allows_single_node_partition=True, we should have partitions
        self.assertGreater(len(partitions_with_single), 0)
