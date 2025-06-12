# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import torch
from exir.backend.canonical_partitioners.group_partitioner import GroupBasedPartitioner

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

        def is_node_supported(self, submodules, node):
            if node.op == "get_attr":
                return True

            if node.name in self.supported_nodes:
                return True

            return False

    def find_nodes_by_names(
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
        return model()

    def create_input(self):
        return torch.randn(5, 10)

    def export_program(self, model, x):
        return torch.export.export(model, (x,))

    def find_input_nodes(self, exported_program, names=None):
        if not names:
            return None
        out = []
        for group in names:
            out.append(self.find_nodes_by_names(group, exported_program.graph_module))
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
