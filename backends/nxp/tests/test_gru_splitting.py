# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.aten_passes.neutron_aten_pass_manager import (
    NeutronAtenPassManager,
)
from executorch.backends.nxp.aten_passes.split_gru_based_on_num_layers import (
    SplitGRUBasedOnNumLayers,
)


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(42)
    np.random.seed(23)


def get_gru_input_shapes(
    input_size=8,
    hidden_size=8,
    sequence_length=8,
    batch_size=8,
    num_layers=1,
    D=1,  # Unidirectional.
):
    input_shapes = [
        (batch_size, sequence_length, input_size),
        (D * num_layers, batch_size, hidden_size),
    ]

    return input_shapes, input_size, hidden_size, sequence_length


class GruModule(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )

    def forward(self, input_, h_0):
        # `input_` has shape
        #   [sequence_length, batch_size] or [sequence_length, batch_size, input_size]  if `batch_first` is False
        #   [batch_size, sequence_length, input_size]  if `batch_first` is True
        # `h_0` has shape [D * num_layers, hidden_size] or [D * num_layers, batch_size, hidden_size]
        #   where `D` is equal to 2 if bidirectional==True otherwise 1
        return self.gru(input_, h_0)


@pytest.mark.parametrize(
    "num_layers", [2, 3, 8], ids=lambda num_layers: f"num_layers = {num_layers}"
)
def test_gru_splitting__with_bias(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(
        num_layers=num_layers
    )
    model = GruModule(input_size, hidden_size, num_layers=num_layers).eval()

    example_input = tuple(torch.ones(input_shape) for input_shape in input_shapes)
    exir_program_aten = torch.export.export(model, example_input).module()

    pre_pass_output = [t.detach() for t in exir_program_aten(*example_input)]
    assert len(exir_program_aten.graph.nodes) == 6 + (num_layers) * 4
    assert (
        len(
            [
                n
                for n in exir_program_aten.graph.nodes
                if n.target == torch.ops.aten.gru.input
            ]
        )
        == 1
    )  # Just 1 `GRU` in the model.

    # Run pre-processing passes of the float32 aten dialect program.
    pytorch_pass_manager = NeutronAtenPassManager([SplitGRUBasedOnNumLayers()])
    pytorch_pass_manager(exir_program_aten)

    post_pass_output = [t.detach() for t in exir_program_aten(*example_input)]
    nodes = list(exir_program_aten.graph.nodes)
    assert len(nodes) == 5 + (num_layers) * 8
    assert nodes[2 + num_layers * 4].target == torch.ops.aten.split.default
    assert (
        len(
            [
                n
                for n in exir_program_aten.graph.nodes
                if n.target == torch.ops.aten.gru.input
            ]
        )
        == num_layers
    )  # Many `GRU` nodes.
    assert nodes[-2].target == torch.ops.aten.cat.default

    assert np.allclose(pre_pass_output[0], post_pass_output[0]), "Main outputs differ"
    assert np.allclose(pre_pass_output[1], post_pass_output[1]), "Hidden outputs differ"


@pytest.mark.parametrize(
    "num_layers", [2, 5], ids=lambda num_layers: f"num_layers = {num_layers}"
)
def test_gru_splitting__no_bias(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(
        num_layers=num_layers
    )
    model = GruModule(input_size, hidden_size, num_layers=num_layers, bias=False).eval()

    example_input = tuple(torch.ones(input_shape) for input_shape in input_shapes)
    exir_program_aten = torch.export.export(model, example_input).module()

    pre_pass_output = [t.detach() for t in exir_program_aten(*example_input)]
    assert len(exir_program_aten.graph.nodes) == 6 + (num_layers) * 2
    assert (
        len(
            [
                n
                for n in exir_program_aten.graph.nodes
                if n.target == torch.ops.aten.gru.input
            ]
        )
        == 1
    )  # Just 1 `GRU` in the model.

    # Run pre-processing passes of the float32 aten dialect program.
    pytorch_pass_manager = NeutronAtenPassManager([SplitGRUBasedOnNumLayers()])
    pytorch_pass_manager(exir_program_aten)

    post_pass_output = [t.detach() for t in exir_program_aten(*example_input)]
    nodes = list(exir_program_aten.graph.nodes)
    assert len(nodes) == 5 + (num_layers) * 6
    assert nodes[2 + num_layers * 2].target == torch.ops.aten.split.default
    assert (
        len(
            [
                n
                for n in exir_program_aten.graph.nodes
                if n.target == torch.ops.aten.gru.input
            ]
        )
        == num_layers
    )  # Many `GRU` nodes.
    assert nodes[-2].target == torch.ops.aten.cat.default

    assert np.allclose(pre_pass_output[0], post_pass_output[0]), "Main outputs differ"
    assert np.allclose(pre_pass_output[1], post_pass_output[1]), "Hidden outputs differ"


@pytest.mark.parametrize(
    "num_layers", [2, 3, 5], ids=lambda num_layers: f"num_layers = {num_layers}"
)
def test_gru_splitting__bidirectional__no_bias(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(
        num_layers=num_layers, D=2
    )
    model = GruModule(
        input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=False
    ).eval()

    example_input = tuple(torch.ones(input_shape) for input_shape in input_shapes)
    exir_program_aten = torch.export.export(model, example_input).module()

    assert len(exir_program_aten.graph.nodes) == 6 + (num_layers) * 4
    assert (
        len(
            [
                n
                for n in exir_program_aten.graph.nodes
                if n.target == torch.ops.aten.gru.input
            ]
        )
        == 1
    )  # Just 1 `GRU` in the model.

    # Run pre-processing passes of the float32 aten dialect program.
    pytorch_pass_manager = NeutronAtenPassManager([SplitGRUBasedOnNumLayers()])
    pytorch_pass_manager(exir_program_aten)

    nodes = list(exir_program_aten.graph.nodes)
    assert len(nodes) == 5 + (num_layers) * 8
    assert nodes[2 + num_layers * 4].target == torch.ops.aten.split.default
    assert (
        len(
            [
                n
                for n in exir_program_aten.graph.nodes
                if n.target == torch.ops.aten.gru.input
            ]
        )
        == num_layers
    )  # Many `GRU` in the model.
    assert nodes[-2].target == torch.ops.aten.cat.default


@pytest.mark.parametrize(
    "num_layers", [2, 3, 7], ids=lambda num_layers: f"num_layers = {num_layers}"
)
def test_gru_splitting__bidirectional__with_bias(num_layers):
    input_shapes, input_size, hidden_size, _ = get_gru_input_shapes(
        num_layers=num_layers, D=2
    )
    model = GruModule(
        input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=True
    ).eval()

    example_input = tuple(torch.ones(input_shape) for input_shape in input_shapes)
    exir_program_aten = torch.export.export(model, example_input).module()

    assert len(exir_program_aten.graph.nodes) == 6 + (num_layers) * 8
    assert (
        len(
            [
                n
                for n in exir_program_aten.graph.nodes
                if n.target == torch.ops.aten.gru.input
            ]
        )
        == 1
    )  # Just 1 `GRU` in the model.

    # Run pre-processing passes of the float32 aten dialect program.
    pytorch_pass_manager = NeutronAtenPassManager([SplitGRUBasedOnNumLayers()])
    pytorch_pass_manager(exir_program_aten)

    nodes = list(exir_program_aten.graph.nodes)
    assert len(nodes) == 5 + (num_layers) * 12
    assert nodes[2 + num_layers * 8].target == torch.ops.aten.split.default
    assert (
        len(
            [
                n
                for n in exir_program_aten.graph.nodes
                if n.target == torch.ops.aten.gru.input
            ]
        )
        == num_layers
    )  # Many `GRU` in the model.
    assert nodes[-2].target == torch.ops.aten.cat.default
