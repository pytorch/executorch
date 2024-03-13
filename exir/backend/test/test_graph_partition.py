# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import unittest
from typing import List, Optional, Tuple

import torch
from executorch import exir
from executorch.exir import CaptureConfig, EdgeCompileConfig
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.node import Node
from torch.fx.passes.operator_support import OperatorSupportBase


class TestGraphPartition(unittest.TestCase):
    def get_graph_module(
        self, module: torch.nn.Module, inputs: Tuple[torch.Tensor]
    ) -> torch.fx.GraphModule:
        capture_config = CaptureConfig()
        graph_module = (
            exir.capture(module, inputs, capture_config)
            .to_edge(
                EdgeCompileConfig(
                    _check_ir_validity=False,
                )
            )
            .exported_program.graph_module
        )

        return graph_module

    def get_node_list(
        self,
        graph_module: torch.fx.GraphModule,
        supported_modules: List[torch.nn.Module],
    ) -> List[List[Node]]:
        pattern_list_map = collections.defaultdict(list)
        for node in graph_module.graph.nodes:
            if "nn_module_stack" in node.meta:
                module_values_list = list(node.meta["nn_module_stack"].values())
                full_qualified_name = module_values_list[-1][0]
                owning_module = module_values_list[-1][1]
                if owning_module in supported_modules:
                    pattern_list_map[(full_qualified_name, owning_module)].append(node)

        return list(pattern_list_map.values())

    def extract_partition_list(
        self,
        graph_module: torch.fx.GraphModule,
        supported_modules: List[torch.nn.Module],
        op_support: Optional[OperatorSupportBase] = None,
    ) -> List:

        node_list = self.get_node_list(graph_module, supported_modules)

        partition_list = generate_partitions_from_list_of_nodes(
            graph_module, node_list, op_support
        )

        return partition_list

    def test_partition_list_without_op_support_one_partition(self):
        """
        check all of submodules should be lowered into a single part
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(32, 32, 1)
                self.conv2 = torch.nn.Conv2d(32, 32, 1)
                self.conv3 = torch.nn.Conv2d(32, 32, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor):
                a = self.conv1(x)
                b = self.conv2(a)
                c = self.conv3(b)
                d = self.conv3(c)
                return self.relu(d)

        example_inputs = (torch.rand(1, 32, 16, 16),)
        test_module = TestModule()
        graph_module = self.get_graph_module(test_module, example_inputs)

        supported_module = [
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.activation.ReLU,
        ]
        partition_list = self.extract_partition_list(graph_module, supported_module)

        assert (
            len(partition_list) == 1
        ), "the subgraph should be divided into a single part"

    def test_partition_list_without_op_support_two_partitions(self):
        """
        check graph will be divided into 2 parts when the supported modules is provided, but OpeartorSupportBase is not provideds
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(32, 32, 1)
                self.conv2 = torch.nn.Conv2d(32, 32, 1)
                self.conv3 = torch.nn.Conv2d(32, 32, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor):
                a = self.conv1(x)
                b = self.conv2(a)
                c = self.conv3(a + b)
                d = self.conv3(c)
                return self.relu(d)

        example_inputs = (torch.rand(1, 32, 16, 16),)
        test_module = TestModule()
        graph_module = self.get_graph_module(test_module, example_inputs)

        supported_module = [
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.activation.ReLU,
        ]
        partition_list = self.extract_partition_list(graph_module, supported_module)

        assert len(partition_list) == 2, "the subgraph should be divided into 2 parts"

        partition_1 = [
            "_param_constant4",
            "_param_constant4_1",
            "_param_constant5",
            "_param_constant5_1",
            "aten_convolution_default_2",
            "aten_convolution_default_3",
            "aten_relu_default",
        ]
        partition_2 = [
            "_param_constant0",
            "_param_constant1",
            "_param_constant2",
            "_param_constant3",
            "aten_convolution_default",
            "aten_convolution_default_1",
        ]

        # extract node names from partition_list, compare them with expected node names
        node_list_1 = []
        for node in partition_list[0].nodes:
            node_list_1.append(node.name)

        node_list_2 = []
        for node in partition_list[1].nodes:
            node_list_2.append(node.name)

        node_list_1 = sorted(node_list_1)
        node_list_2 = sorted(node_list_2)

        self.assertEqual(node_list_1, partition_1)
        self.assertEqual(node_list_2, partition_2)

    def test_graph_partition_with_op_support(self):
        """
        check graph will be divided into 2 parts when the supported modules and OpeartorSupportBase are provided,
        """

        class TestOperatorSupport(OperatorSupportBase):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return node.op == "call_function" and node.target in [
                    exir_ops.edge.aten.div.Tensor,
                    exir_ops.edge.aten.add.Tensor,
                ]

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(32, 32, 1)
                self.conv2 = torch.nn.Conv2d(32, 32, 1)
                self.conv3 = torch.nn.Conv2d(32, 32, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor):
                a = self.conv1(x)
                b = self.conv2(a)
                c = self.conv3(a + b)
                d = self.conv3(c)
                c, _ = torch.max(c, dim=2)
                d, _ = torch.max(d, dim=2)
                e = d - c
                return self.relu(e)

        example_inputs = (torch.rand(1, 32, 16, 16),)
        test_module = TestModule()
        graph_module = self.get_graph_module(test_module, example_inputs)

        supported_module = [
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.activation.ReLU,
        ]
        partition_list = self.extract_partition_list(
            graph_module, supported_module, TestOperatorSupport()
        )

        assert len(partition_list) == 2, "the subgraph should be divided into 2 parts"

        partition_1 = ["aten_relu_default"]
        partition_2 = [
            "_param_constant0",
            "_param_constant1",
            "_param_constant2",
            "_param_constant3",
            "_param_constant4",
            "_param_constant4_1",
            "_param_constant5",
            "_param_constant5_1",
            "aten_add_tensor",
            "aten_convolution_default",
            "aten_convolution_default_1",
            "aten_convolution_default_2",
            "aten_convolution_default_3",
        ]

        # extract node names from partition_list, compare them with expected node names
        node_list_1 = []
        for node in partition_list[0].nodes:
            node_list_1.append(node.name)

        node_list_2 = []
        for node in partition_list[1].nodes:
            node_list_2.append(node.name)

        node_list_1 = sorted(node_list_1)
        node_list_2 = sorted(node_list_2)

        self.assertEqual(node_list_1, partition_1)
        self.assertEqual(node_list_2, partition_2)
