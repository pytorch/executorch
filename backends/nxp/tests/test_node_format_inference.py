# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch import exir
from executorch.backends.nxp.backend.node_format_inference import (
    NodeFormat,
    NodeFormatInference,
)
from executorch.backends.nxp.neutron_pass_manager import NeutronPassManager
from executorch.backends.nxp.tests.models import (
    Conv2dModule,
    MaxPool2dModule,
    SoftmaxModule,
)
from executorch.backends.xnnpack._passes import RemoveGetItemPass
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier


def test_convolution():
    model = Conv2dModule()
    example_input = (torch.ones(1, 4, 32, 32),)

    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    node_formats = NodeFormatInference(edge_program).identify_node_formats()

    expected_mapping = {
        "p_conv_weight": NodeFormat.CHANNELS_FIRST,
        "p_conv_bias": NodeFormat.FORMATLESS,
        "x": NodeFormat.CHANNELS_FIRST,
        "aten_convolution_default": NodeFormat.CHANNELS_FIRST,
        "output": NodeFormat.CHANNELS_FIRST,
    }

    for node, node_format in node_formats.items():
        assert expected_mapping[node.name] == node_format


def test_softmax():
    model = SoftmaxModule(1)
    example_input = (torch.ones(1, 4, 32, 32),)

    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    node_formats = NodeFormatInference(edge_program).identify_node_formats()

    expected_mapping = {
        "x": NodeFormat.FORMATLESS,
        "aten__softmax_default": NodeFormat.FORMATLESS,
        "output": NodeFormat.FORMATLESS,
    }

    for node, node_format in node_formats.items():
        assert expected_mapping[node.name] == node_format


def test_maxpool2d():
    model = MaxPool2dModule()
    example_input = (torch.ones(1, 4, 32, 32),)

    exir_program = torch.export.export(model, example_input)
    edge_program = exir.to_edge(exir_program).exported_program()

    # We need to create custom model verifier with max_pool2d added as exception.
    # Otherwise, we get violation that this op is not part of ATen Core ops.
    edge_program._verifiers = [
        EXIREdgeDialectVerifier(
            class_only=True,
            core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
        )
    ]

    # Remove MaxPool-related "getitem" nodes from graph
    edge_program = NeutronPassManager(edge_program, [RemoveGetItemPass]).transform()
    node_formats = NodeFormatInference(edge_program).identify_node_formats()

    expected_mapping = {
        "x": NodeFormat.CHANNELS_FIRST,
        "aten_max_pool2d_default": NodeFormat.CHANNELS_FIRST,
        "output": NodeFormat.CHANNELS_FIRST,
    }

    for node, node_format in node_formats.items():
        assert expected_mapping[node.name] == node_format
