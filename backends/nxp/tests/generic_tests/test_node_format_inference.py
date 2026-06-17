# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from executorch import exir
from executorch.backends.nxp.backend.node_format_inference import (
    DataFormat,
    NodeFormatInference,
    NXP_NODE_FORMAT,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops

from executorch.backends.nxp.tests.models import (
    Conv2dModule,
    MaxPool2dModule,
    SoftmaxModule,
)
from executorch.backends.nxp.tests.ops_aliases import (
    ExecutorchDelegateCall,
    MaxPool2DWithIndices,
)


def test_convolution():
    model = Conv2dModule()
    example_input = (torch.ones(1, 4, 32, 32),)

    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    NodeFormatInference(edge_program).identify_node_formats()

    expected_mapping = {
        "p_conv_weight": DataFormat.CHANNELS_FIRST,
        "p_conv_bias": DataFormat.FORMATLESS,
        "x": DataFormat.CHANNELS_FIRST,
        "aten_convolution_default": DataFormat.CHANNELS_FIRST,
        "output": DataFormat.CHANNELS_FIRST,
    }

    for node in edge_program.graph.nodes:
        assert expected_mapping[node.name] == node.meta[NXP_NODE_FORMAT]


def test_softmax():
    model = SoftmaxModule(1)
    example_input = (torch.ones(1, 4, 32, 32),)

    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    NodeFormatInference(edge_program).identify_node_formats()

    expected_mapping = {
        "x": DataFormat.FORMATLESS,
        "aten__softmax_default": DataFormat.FORMATLESS,
        "output": DataFormat.FORMATLESS,
    }

    for node in edge_program.graph.nodes:
        assert expected_mapping[node.name] == node.meta[NXP_NODE_FORMAT]


def test_max_pool2d():
    model = MaxPool2dModule()
    example_input = (torch.ones(1, 4, 32, 32),)

    exir_program = torch.export.export(model, example_input)
    epm = exir.to_edge(exir_program)

    NodeFormatInference(epm.exported_program()).identify_node_formats()

    expected_mapping = {
        "x": DataFormat.CHANNELS_FIRST,
        "aten_max_pool2d_with_indices_default": DataFormat.CHANNELS_FIRST,
        "getitem": DataFormat.CHANNELS_FIRST,
        "output": DataFormat.CHANNELS_FIRST,
    }

    for node in epm.exported_program().graph.nodes:
        assert expected_mapping[node.name] == node.meta[NXP_NODE_FORMAT]


def test_unhandled_channels_first_node(caplog):
    # This test focuses on the case where some operator requires the channels first format, which is enforced in the
    #  `NodeConverter`, but the `NodeFormatInference` fails to reflect this.
    # We use the `MaxPool` operator for this, and we temporarily modify the `NodeFormatInference` to trigger the issue.

    model = MaxPool2dModule()
    input_shape = (1, 4, 32, 32)

    # Temporarily "break" the NodeFormatInference.
    old_channels_first_ops = NodeFormatInference.ops_with_channels_first_nodes
    NodeFormatInference.ops_with_channels_first_nodes = {}

    with caplog.at_level(
        logging.WARNING,
        logger="executorch.backends.nxp.backend.ir.converter.node_converter",
    ):
        ep = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure the `MaxPool` wasn't delegated.
    assert graph_contains_any_of_ops(ep.graph, [MaxPool2DWithIndices])
    assert not graph_contains_any_of_ops(ep.graph, [ExecutorchDelegateCall])

    # Make sure the warning is printed.
    assert any(
        "`aten_max_pool2d_with_indices_default` requires channels-first format for its input and output, but the "
        "inferred format does not satisfy this requirement" in message
        for message in caplog.messages
    )

    # Restore the original channels first ops configuration.
    NodeFormatInference.ops_with_channels_first_nodes = old_channels_first_ops
