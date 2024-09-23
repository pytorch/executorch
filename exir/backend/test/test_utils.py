# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch import exir
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.backend.utils import (
    format_delegated_graph,
    get_delegates,
    get_non_lowered_nodes,
    is_identical_graph,
)

from executorch.exir.dialects._ops import bind_pattern_to_op, ops as exir_ops
from torch.export import export, ExportedProgram
from torch.fx import symbolic_trace
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.library import Library

T_QuantPerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
T_DQuantPerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


class TestUtils(unittest.TestCase):
    def test_identical_graph_with_unused_args(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # y is not used arg
                return x

        m = MyModule()
        graph_module: torch.fx.GraphModule = symbolic_trace(m)
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_with_used_args(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x, y

        m = MyModule()
        graph_module: torch.fx.GraphModule = symbolic_trace(m)
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_for_linear(self):
        graph_module: torch.fx.GraphModule = symbolic_trace(torch.nn.Linear(10, 10))
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_for_composite_module(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        graph_module: torch.fx.GraphModule = symbolic_trace(MyModule())
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_not_identical_graph_for_args(self):
        class MyModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # y is not used arg
                return x + 1

        class MyModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + 1, y + 2

        graph_module_1: torch.fx.GraphModule = (
            to_edge(
                export(
                    MyModule1(),
                    (torch.rand(3, 4), torch.rand(3, 4)),
                )
            )
            .exported_program()
            .graph_module
        )
        graph_module_2: torch.fx.GraphModule = (
            to_edge(
                export(
                    MyModule2(),
                    (torch.rand(3, 4), torch.rand(3, 4)),
                )
            )
            .exported_program()
            .graph_module
        )
        is_matched = is_identical_graph(graph_module_1, graph_module_2)
        self.assertFalse(is_matched)

    def test_match_attrs(self):
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weght = torch.nn.Parameter(torch.ones(3, 3))
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                a = x + self.weght
                b = self.linear(x)
                return a, b

        inputs = (torch.ones(3, 3),)

        large_model = (
            to_edge(
                export(
                    LargeModel(),
                    inputs,
                ),
                compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
            )
            .exported_program()
            .graph_module
        )

        pattern = (
            to_edge(
                export(torch.nn.Linear(3, 3), inputs),
                compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
            )
            .exported_program()
            .graph_module.graph
        )

        subgraph_matcher = SubgraphMatcher(pattern)
        match_result = subgraph_matcher.match(large_model.graph)

        # Should find exact one match
        self.assertEqual(len(match_result), 1)

    def test_invalid_partitioner_without_partitioner(self):
        """
        Tests replacing literals with placeholders in the case there are
        `getitem` calls which do not have a schema.
        """

        class InvalidPartitioner(Partitioner):
            """
            Partitions all add/mul nodes regardless of order
            """

            def __init__(self) -> None:
                # A valid partitioner should have partition_tags
                self.test = "a"

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                return PartitionResult(
                    tagged_exported_program=edge_exported_program, partition_tags=None
                )

        exported_program = to_edge(
            export(
                torch.nn.Linear(3, 3),
                (torch.randn(3, 3),),
            )
        )

        error_msg = r"needs a `partition_tags` field containing a mapping of tags to delegate spec"
        with self.assertRaisesRegex(
            AssertionError,
            error_msg,
        ):
            _ = to_backend(exported_program.exported_program(), InvalidPartitioner())

    test_lib = Library("test_lib", "DEF")

    @staticmethod
    @bind_pattern_to_op(
        test_lib, "test_q_linear(Tensor x, Tensor weight, Tensor bias) -> Tensor"
    )
    def q_linear(x, weight, bias):
        return x

    def test_get_non_lowered_nodes(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        edge = to_edge(export(m, inputs))
        edge = edge.to_backend(AddMulPartitionerDemo())
        number_of_cpu_nodes = get_non_lowered_nodes(edge.exported_program().graph)
        # Only sub is not not lowerable
        self.assertEqual(len(number_of_cpu_nodes), 1)

    def test_get_delegates(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        edge = to_edge(export(m, inputs))
        edge = edge.to_backend(AddMulPartitionerDemo())
        number_of_delegates = get_delegates(edge.exported_program().graph)
        # there will be 2 delegates: (mm + add) -> sub -> (mm + add)
        self.assertEqual(len(number_of_delegates), 2)

    def test_print_delegted_graph(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        edge = to_edge(export(m, inputs)).to_backend(AddMulPartitionerDemo())

        graph_str = format_delegated_graph(edge.exported_program().graph_module)
        self.assertIn(
            "BackendWithCompilerDemo",
            graph_str,
            "Expect to find the backend id in the graph format string",
        )
        self.assertIn(
            "executorch.exir.dialects.edge._ops.aten.mm.default",
            graph_str,
            "Expect to see the aten.mm in the delegated graph",
        )
