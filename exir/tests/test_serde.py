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
from executorch.backends.backend_api import CompileSpec, to_backend
from executorch.backends.test.backend_with_compiler_demo import (  # noqa
    BackendWithCompilerDemo,
)
from executorch.backends.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir import EdgeCompileConfig
from executorch.exir.serde.serialize import deserialize, serialize
from torch._export.exported_program import ExportedProgram as TorchExportedProgram
from torch.utils import _pytree as pytree


# Tests for serializing to json and back
class TestSerde(unittest.TestCase):
    def setUp(self) -> None:
        # TODO(gasoon): Remove this once serde is fully migrated to Edge ops
        self.edge_complie_config = EdgeCompileConfig(_use_edge_ops=False)

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

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
            self.assertTrue(torch.allclose(orig, loaded))

    # pyre-ignore
    def check_serde(self, m, inputs) -> None:
        aten = exir.capture(m, inputs, exir.CaptureConfig(pt2_mode=True))
        aten_new = deserialize(*serialize(aten.exported_program))
        self.check_ep(aten.exported_program, aten_new, inputs)

        edge = aten.to_edge(self.edge_complie_config)
        edge_new = deserialize(*serialize(edge.exported_program))
        self.check_ep(edge.exported_program, edge_new, inputs)

        executorch = edge.to_executorch().dump_exported_program()
        executorch_new = deserialize(*serialize(executorch))
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
        edgeir_m = exir.capture(
            sin_module, model_inputs, exir.CaptureConfig(pt2_mode=True)
        ).to_edge(self.edge_complie_config)
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_sin_module = to_backend(
            "BackendWithCompilerDemo", edgeir_m.exported_program, compile_specs
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

        aten = exir.capture(
            composite_model, model_inputs, exir.CaptureConfig(pt2_mode=True)
        )
        aten_new = deserialize(*serialize(aten.exported_program))
        self.check_ep(aten.exported_program, aten_new, model_inputs)

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

        ep = exir.capture(m, inputs, exir.CaptureConfig(pt2_mode=True)).to_edge(
            self.edge_complie_config
        )
        edge = to_backend(ep.exported_program, AddMulPartitionerDemo)
        edge_new = deserialize(*serialize(edge))
        self.check_ep(edge, edge_new, inputs)
