# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Tuple

import executorch.exir as exir

import torch
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import CompileSpec, to_backend
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)

from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.serde.serialize import deserialize, serialize
from torch import nn
from torch._export.exported_program import ExportedProgram as TorchExportedProgram
from torch.export import export
from torch.utils import _pytree as pytree


# Tests for serializing to json and back
class TestSerde(unittest.TestCase):
    def check_ep(
        self,
        ep1: TorchExportedProgram,
        ep2: TorchExportedProgram,
        inputs: Tuple[exir.Value, ...],
    ) -> None:
        """
        Checks if two graphs are equivalent
        """
        orig_outputs = ep1(*inputs)
        loaded_outputs = ep2(*inputs)

        flat_orig_outputs, _ = pytree.tree_flatten(orig_outputs)
        flat_loaded_outputs, _ = pytree.tree_flatten(loaded_outputs)

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs, strict=True):
            self.assertTrue(torch.allclose(orig, loaded))

    # pyre-ignore
    def check_serde(self, m, inputs) -> None:
        aten = export(m, inputs)
        aten_new = deserialize(serialize(aten))
        self.check_ep(aten, aten_new, inputs)

        edge = to_edge(aten)
        edge_new = deserialize(serialize(edge.exported_program()))
        self.check_ep(edge.exported_program(), edge_new, inputs)

        executorch = edge.to_executorch().exported_program()
        executorch_new = deserialize(serialize(executorch))
        with torch.no_grad():
            self.check_ep(executorch, executorch_new, inputs)

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
        self.check_serde(model, inputs)

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
        edgeir_m = to_edge(export(sin_module, model_inputs))
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

        edge = to_edge(export(composite_model, model_inputs))
        edge_new = deserialize(serialize(edge.exported_program()))
        self.check_ep(edge.exported_program(), edge_new, model_inputs)

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

        ep = to_edge(export(m, inputs))
        edge = ep.to_backend(AddMulPartitionerDemo())
        edge_new = deserialize(serialize(edge.exported_program()))
        self.check_ep(edge.exported_program(), edge_new, inputs)

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
        edge = to_edge(export(m, inputs))
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
