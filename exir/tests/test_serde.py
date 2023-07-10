# pyre-strict

import unittest
from typing import Tuple

import executorch.exir as exir

import torch
from executorch.exir.serde.serialize import deserialize, serialize
from torch._export.exported_program import ExportedProgram as TorchExportedProgram
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

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
            self.assertTrue(torch.allclose(orig, loaded))

    # pyre-ignore
    def check_serde(self, m, inputs) -> None:
        aten = exir.capture(m, inputs, exir.CaptureConfig(pt2_mode=True))
        aten_new = deserialize(*serialize(aten))
        self.check_ep(aten, aten_new, inputs)

        edge = aten.to_edge()
        edge_new = deserialize(*serialize(edge))
        self.check_ep(edge, edge_new, inputs)

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
