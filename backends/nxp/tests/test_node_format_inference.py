# Copyright 2024-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch import exir
from executorch.backends.nxp.backend.node_format_inference import (
    DataFormat,
    NodeFormatInference,
    NXP_NODE_FORMAT,
)
from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
    NeutronEdgePassManager,
)
from executorch.backends.nxp.tests.models import (
    Conv2dModule,
    MaxPool2dModule,
    SoftmaxModule,
)
from executorch.exir import EdgeCompileConfig


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

    # We need to add the  `aten.max_pool2d.default` as an exception, otherwise we would get violation that this op is
    #  not part of ATen Core ops.
    exception_list = [torch.ops.aten.max_pool2d.default]
    epm = exir.to_edge(
        exir_program,
        compile_config=EdgeCompileConfig(_core_aten_ops_exception_list=exception_list),
    )

    epm = epm.transform(NeutronEdgePassManager())
    NodeFormatInference(epm.exported_program()).identify_node_formats()

    expected_mapping = {
        "x": DataFormat.CHANNELS_FIRST,
        "aten_max_pool2d_default": DataFormat.CHANNELS_FIRST,
        "output": DataFormat.CHANNELS_FIRST,
    }

    for node in epm.exported_program().graph.nodes:
        assert expected_mapping[node.name] == node.meta[NXP_NODE_FORMAT]
