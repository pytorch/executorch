# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import pytest
import torch

from executorch.backends.nxp.tests.executorch_pipeline import _quantize_model
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


class DeadCodeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eval()

    def forward(self, x):
        _ = torch.add(x, x)  # Dead code
        return torch.mul(x, x)


class TestRemovingDeadCode(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests

    def test_removing_dead_code(self):
        input_shape = (42,)
        example_inputs = (torch.ones(input_shape),)
        model = DeadCodeModule()

        exir_program_aten = torch.export.export(model, example_inputs, strict=True)

        # Make sure the model contains the dead code.
        assert graph_contains_any_of_ops(
            exir_program_aten.module().graph, [torch.ops.aten.add.Tensor]
        )

        # The `NeutronQuantizer` should remove the dead code in the `transform_for_annotation()` method.
        exir_program_aten_quant = _quantize_model(
            exir_program_aten.module(), [example_inputs]
        )

        # Make sure the is no `add` operation in the graph anymore.
        assert not any(
            "add" in str(node.target) for node in exir_program_aten_quant.graph.nodes
        )

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(23)
