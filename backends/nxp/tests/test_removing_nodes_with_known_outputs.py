# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.aten_passes.remove_nodes_with_known_outputs import (
    RemoveNodesWithKnownOutputs,
)
from executorch.backends.nxp.aten_passes.split_gru_based_on_num_layers import (
    SplitGRUBasedOnNumLayers,
)
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from parameterized import parameterized
from torch import nn


class GRUModel(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.gru = torch.nn.GRU(8, 8, num_layers=num_layers)

    def forward(self, input_):
        # `input_` has shape [sequence_length, batch_size, input_size] ([8, 1, 8])
        return self.gru(
            input_, None
        )  # The initial hidden is `None`, which will result in a `Zeros` node being added.


class TestRemovingNodesWithKnownOutputs(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests.

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    def test_removing_nodes__zeros(self):
        model = GRUModel()

        input_shape = (8, 1, 8)
        example_input = (torch.ones(input_shape),)

        exir_program_aten = torch.export.export(model, example_input).module()

        # Make sure the `aten.zeros` is in the model.
        assert graph_contains_any_of_ops(
            exir_program_aten.graph, [torch.ops.aten.zeros.default]
        )
        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Apply the optimization.
        NeutronAtenPassManager([RemoveNodesWithKnownOutputs()])(exir_program_aten)

        # Make sure the `aten.zeros` is no longer in the model.
        assert not graph_contains_any_of_ops(
            exir_program_aten.graph, [torch.ops.aten.zeros.default]
        )
        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])
        assert np.allclose(outputs_before[1], outputs_after[1])

    @parameterized.expand([2, 8])
    def test_removing_nodes__split(self, num_layers):
        # `num_layers > 1` will result in a `split` operator being added. It's input will be a `zeros` operator, which
        #  provides the static 0s input data.
        model = GRUModel(num_layers).eval()

        input_shape = (8, 1, 8)
        example_input = (torch.ones(input_shape),)

        exir_program_aten = torch.export.export(model, example_input).module()

        # Apply the pass to split the `aten.gru.input` into multiple instances, and add a `split` node.
        NeutronAtenPassManager([SplitGRUBasedOnNumLayers()])(exir_program_aten)

        # Make sure the `aten.zeros` and `torch.split` are in the model.
        assert graph_contains_any_of_ops(
            exir_program_aten.graph, [torch.ops.aten.zeros.default]
        )
        assert graph_contains_any_of_ops(
            exir_program_aten.graph, [torch.ops.aten.split.default]
        )
        outputs_before = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Apply the optimization.
        NeutronAtenPassManager([RemoveNodesWithKnownOutputs()])(exir_program_aten)

        # Make sure the `aten.zeros` and `torch.split` are no longer in the model.
        assert not graph_contains_any_of_ops(
            exir_program_aten.graph, [torch.ops.aten.zeros.default]
        )
        assert not graph_contains_any_of_ops(
            exir_program_aten.graph, [torch.ops.aten.split.default]
        )
        outputs_after = [o.detach().numpy() for o in exir_program_aten(*example_input)]

        # Make sure the model still produces the exact same output.
        assert np.allclose(outputs_before[0], outputs_after[0])
        assert np.allclose(outputs_before[1], outputs_after[1])
