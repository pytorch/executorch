# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import tempfile
import unittest
from typing import Tuple

import executorch.exir as exir

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
)
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import CompileSpec, to_backend
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)

from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.program._program import (
    EdgeProgramManager,
    to_edge_transform_and_lower,
)
from executorch.exir.serde.serialize import deserialize, serialize
from torch import nn
from torch.export import export
from torch.export.exported_program import ExportedProgram as TorchExportedProgram
from torch.utils import _pytree as pytree


# Tests for serializing to json and back
class TestSerde(unittest.TestCase):
    def check_ep(
        self,
        ep1: TorchExportedProgram,
        ep2: TorchExportedProgram,
        inputs: Tuple[exir.Value, ...],
        compare_closeness: bool = False,
    ) -> None:
        """
        Checks if two graphs are equivalent
        """
        orig_outputs = ep1.module()(*inputs)
        loaded_outputs = ep2.module()(*inputs)

        flat_orig_outputs, _ = pytree.tree_flatten(orig_outputs)
        flat_loaded_outputs, _ = pytree.tree_flatten(loaded_outputs)

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs, strict=True):
            self.assertTrue(torch.allclose(orig, loaded))

        if compare_closeness:
            self.assertEqual(len(ep1.graph.nodes), len(ep2.graph.nodes))
            for node_a, node_b in zip(ep1.graph.nodes, ep2.graph.nodes):
                self.assertEqual(node_a.target, node_b.target)
                self.assertEqual(node_a.name, node_b.name)
                self.assertEqual(node_a.type, node_b.type)
                self.assertEqual(node_a.op, node_b.op)
                if node_a.op != "call_function":
                    continue

                self.assertEqual(
                    node_a.meta.get("debug_handle"), node_b.meta.get("debug_handle")
                )
                from_node_a = node_a.meta.get("from_node")
                from_node_b = node_b.meta.get("from_node")

                if from_node_a is None:
                    self.assertIsNone(from_node_b)
                else:
                    self.assertIsNotNone(from_node_b)
                    for node_source_a, node_source_b in zip(from_node_a, from_node_b):
                        self.assertEqual(
                            node_source_a.to_dict(), node_source_b.to_dict()
                        )

    # pyre-ignore
    def check_serde(self, m, inputs, check_executorch=True) -> None:
        aten = export(m, inputs, strict=True)
        aten_new = deserialize(serialize(aten))
        self.check_ep(aten, aten_new, inputs, compare_closeness=True)

        edge = to_edge(aten)
        edge_new = deserialize(serialize(edge.exported_program()))
        self.check_ep(edge.exported_program(), edge_new, inputs, compare_closeness=True)

        buffer = io.BytesIO()
        exir.save(edge.exported_program(), buffer)
        buffer.seek(0)
        loaded_ep = exir.load(buffer)
        self.check_ep(edge.exported_program(), loaded_ep, inputs)

        executorch = edge.to_executorch().exported_program()
        executorch_new = deserialize(serialize(executorch))
        if check_executorch:
            with torch.no_grad():
                self.check_ep(executorch, executorch_new, inputs)

                buffer = io.BytesIO()
                exir.save(executorch, buffer)
                buffer.seek(0)
                loaded_ep = exir.load(buffer)
                self.check_ep(executorch, loaded_ep, inputs)

    def test_basic(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + x
                x = x * x
                x = x / x
                return x, x.clone()

        inputs = (torch.ones([512], requires_grad=True),)
        self.check_serde(MyModule(), inputs)

    def test_to_out_variant_singleon_tensor_list(self) -> None:
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.split(x, 10)

            def get_random_inputs(self):
                return (torch.randn(10),)

        model = MyModel()
        inputs = model.get_random_inputs()
        # We set check_executorch to false for this test because this triggers
        # an edge case where calling .module() on the executorch exported program
        # will cause an unlift pass to be run on the graph and dead code elimination
        # will be subsequently run, which essentially causes the split_copy op to be
        # removed.
        self.check_serde(model, inputs, check_executorch=False)

    def test_to_out_variant_multiple_out(self) -> None:
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                values, indices = torch.topk(x, 5)
                return (values, indices)

            def get_random_inputs(self):
                return (torch.randn(10),)

        model = MyModel()
        inputs = model.get_random_inputs()
        self.check_serde(model, inputs)

    def test_delegate(self) -> None:
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        sin_module = SinModule()
        model_inputs = (torch.ones(1),)
        edgeir_m = to_edge(export(sin_module, model_inputs, strict=True))
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_sin_module = to_backend(
            BackendWithCompilerDemo.__name__, edgeir_m.exported_program(), compile_specs
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_linear_sin = lowered_sin_module

            def forward(self, x):
                return self.lowered_linear_sin(x)

        composite_model = CompositeModule()
        model_inputs = (torch.ones(1),)

        composite_model(*model_inputs)

        edge = to_edge(export(composite_model, model_inputs, strict=True))
        edge_new = deserialize(serialize(edge.exported_program()))
        self.check_ep(edge.exported_program(), edge_new, model_inputs)

    def test_model_with_weights(self) -> None:
        class LinearAdd(nn.Module):
            def __init__(self, M: int, N: int):
                super().__init__()
                self.M = M
                self.N = N
                self.linear = torch.nn.Linear(M, N)

            def forward(self, x, y):
                x = self.linear(x)
                y = self.linear(y)
                return torch.add(x, y)

            @classmethod
            def _get_random_inputs(cls):
                return (torch.rand(128, 20), torch.rand(128, 20))

        linear_add = LinearAdd(20, 30)
        model_inputs = LinearAdd._get_random_inputs()

        self.check_serde(linear_add, model_inputs)

    def test_delegate_partitioner(self) -> None:
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

        ep = to_edge(export(m, inputs, strict=True))
        edge = ep.to_backend(AddMulPartitionerDemo())
        edge_new = deserialize(serialize(edge.exported_program()))
        self.check_ep(edge.exported_program(), edge_new, inputs)

    def test_delegate_xnnpack(self) -> None:
        class SimpleConv1DModel(nn.Module):
            def __init__(self):
                super(SimpleConv1DModel, self).__init__()
                self.conv1 = nn.Conv1d(
                    in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                x = self.conv1(x)
                return x

        x = torch.randn(64, 1, 100)
        model = SimpleConv1DModel()
        ep = torch.export.export(model, (x,))
        edge_orig = to_edge_transform_and_lower(
            ep, partitioner=[XnnpackFloatingPointPartitioner()]
        )

        with tempfile.NamedTemporaryFile() as f:
            exir.save(edge_orig.exported_program(), f)
            edge_deserialized = EdgeProgramManager(exir.load(f))
            self.assertTrue(
                edge_orig.to_executorch().buffer
                == edge_deserialized.to_executorch().buffer
            )

    def test_meta_stack_trace_module_hierarchy(self) -> None:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv_layer = nn.Conv2d(
                    in_channels=1, out_channels=64, kernel_size=3, padding=1
                )

            def forward(self, x):
                return self.conv_layer(x)

        m = Model()
        inputs = (torch.randn(1, 1, 32, 32),)

        metadata = ()
        edge = to_edge(export(m, inputs, strict=True))
        for node in edge.exported_program().graph_module.graph.nodes:
            if "convolution" in str(node.target):
                metadata = (
                    node.meta.get("stack_trace"),
                    node.meta.get("nn_module_stack"),
                )

        metadata_serde = ()
        edge_new = deserialize(serialize(edge.exported_program()))
        for node in edge_new.graph_module.graph.nodes:
            if "convolution" in str(node.target):
                metadata_serde = (
                    node.meta.get("stack_trace"),
                    node.meta.get("nn_module_stack"),
                )
        self.assertTrue(len(metadata) != 0 and len(metadata_serde) != 0)
        self.assertTrue(
            all(val is not None for val in metadata)
            and all(val is not None for val in metadata_serde)
        )
        self.assertEqual(metadata[0], metadata_serde[0])
        self.assertEqual(list(metadata[1].keys()), list(metadata_serde[1].keys()))

    def test_meta_debug_handle_and_from_node(self) -> None:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv_layer = nn.Conv2d(
                    in_channels=1, out_channels=64, kernel_size=3, padding=1
                )

            def forward(self, x):
                return self.conv_layer(x)

        m = Model()
        inputs = (torch.randn(1, 1, 32, 32),)

        edge = to_edge(export(m, inputs, strict=True))
        edge_new = deserialize(serialize(edge.exported_program()))
        for node, node_new in zip(
            edge.exported_program().graph_module.graph.nodes,
            edge_new.graph_module.graph.nodes,
        ):
            if node.op not in {"placeholder", "output"}:
                self.assertIsNotNone(node.meta.get("debug_handle"))
                self.assertIsNotNone(node.meta.get("from_node"))
                self.assertEqual(
                    node.meta.get("debug_handle"), node_new.meta.get("debug_handle")
                )
                self.assertEqual(
                    len(node.meta.get("from_node")), len(node_new.meta.get("from_node"))
                )
                for node_source, node_source_new in zip(
                    node.meta.get("from_node"), node_new.meta.get("from_node")
                ):
                    self.assertEqual(node_source.to_dict(), node_source_new.to_dict())

    def test_memory_ops(self) -> None:
        class MemoryOpsModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = exir.memory.view(x, (10, 10))
                return x + y

        inputs = (
            torch.randn(100),
            torch.randn(10, 10),
        )
        self.check_serde(MemoryOpsModule(), inputs)
