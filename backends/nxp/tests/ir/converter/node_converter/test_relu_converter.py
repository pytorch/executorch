# Copyright 2024,2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
    exir_ops,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    to_edge_program,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.backends.nxp.tests.graph_verifier import BaseGraphVerifier
from executorch.backends.nxp.tests.models import Conv2dModule, LinearModule, ReLUModule
from executorch.backends.nxp.tests.nsys_testing import lower_run_compare
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


ExecutorchDelegateCall = torch.ops.higher_order.executorch_call_delegate
ReLU = exir_ops.edge.aten.relu.default


class ConvReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = Conv2dModule()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


class LinearReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = LinearModule(bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)


def test_relu_conversion():
    input_shape = (10, 4, 32, 32)
    edge_program = to_edge_program(ReLUModule(), input_shape).exported_program()

    input_data = 2 * np.random.random(input_shape).astype(np.float32) - 1

    convert_run_compare(edge_program, input_data=input_data)


def test_relu_with_conv_quant_conversion(mocker, use_qat):
    input_shape = (1, 4, 32, 32)
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    delegated_ep = to_quantized_edge_program(
        ConvReLUModule(),
        input_shape,
        use_qat=use_qat,
        use_neutron_for_format_conversion=False,
    ).exported_program()

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Capture converted program
    edge_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (
        (2 * np.random.random(input_shape).astype(np.float32) - 1) * 50
    ).astype(np.int8)

    # Make sure the `relu` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ReLU])

    convert_run_compare(
        edge_program,
        input_data,
        tfl_model=tflite_flatbuffers_model,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
    )


def test_relu_with_linear_quant_conversion(mocker, use_qat):
    input_shape = (256, 32)
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    delegated_ep = to_quantized_edge_program(
        LinearReLUModule(), input_shape, use_qat=use_qat
    ).exported_program()

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Capture converted program
    edge_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (
        (2 * np.random.random(input_shape).astype(np.float32) - 1) * 50
    ).astype(np.int8)

    # Make sure the `relu` was delegated.
    assert graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ReLU])

    convert_run_compare(edge_program, input_data, tfl_model=tflite_flatbuffers_model)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param(
            (3, 9, 7), id="num_channels not divisible by NUM_MACS, alone in partition"
        ),
    ],
)
def test_relu_conversion__unsupported(mocker, input_shape):
    delegated_ep = to_quantized_edge_program(
        ReLUModule(), input_shape
    ).exported_program()

    # Make sure the `relu` was NOT delegated.
    assert not graph_contains_any_of_ops(delegated_ep.graph, [ExecutorchDelegateCall])
    assert graph_contains_any_of_ops(delegated_ep.graph, [ReLU])


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param(
            (3, 9, 7), id="num_channels not divisible by NUM_MACS, alone in partition"
        ),
    ],
)
def test_relu_conversion__new_flow_support(mocker, input_shape):
    model = ReLUModule()
    graph_verifier = BaseGraphVerifier(
        exp_num_delegate_call_nodes=1,  # Delegated AvgPool.
        exp_non_delegated_nodes=[],
    )

    lower_run_compare(model, input_shape, graph_verifier, use_new_flow_neutron_c=True)
