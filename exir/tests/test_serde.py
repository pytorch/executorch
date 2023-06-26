# pyre-strict

import unittest
from typing import Tuple

import executorch.exir as exir

import torch
from executorch.exir.serde.serialize import deserialize, serialize
from torch.utils import _pytree as pytree


# Tests for serializing to json and back
class TestSerde(unittest.TestCase):
    def check_ep(
        self,
        ep1: exir.ExportedProgram,
        ep2: exir.ExportedProgram,
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
        aten = exir.capture(MyModule(), inputs, exir.CaptureConfig(pt2_mode=True))
        aten_new = deserialize(*serialize(aten))
        self.check_ep(aten, aten_new, inputs)
