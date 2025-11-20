# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import (
    AddTensorConvModule,
    AddTensorModule,
    AddTensorOneInputModule,
)
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((4,), id="1D."),
        pytest.param((6, 6), id="2D."),
        pytest.param((1, 4, 8), id="3D."),
        pytest.param((1, 4, 8, 8), id="4D."),
    ],
)
def test_add_tensor_quant_conversion(mocker, input_shape):
    model = AddTensorModule()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, [input_shape, input_shape])

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)
    input_data = {0: input_data, 1: input_data}

    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((4,), id="1D."),
        pytest.param((6, 6), id="2D."),
        pytest.param((1, 4, 8), id="3D."),
        pytest.param((1, 4, 8, 8), id="4D."),
    ],
)
def test_add_tensor_one_input_quant_conversion(mocker, input_shape):
    model = AddTensorOneInputModule()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 4, 8, 8), id="4D."),
        pytest.param((1, 4, 5, 5), id="4D, product of dims is not a multiple of 8."),
    ],
)
def test_add_tensor_w_conv_quant_conversion(mocker, input_shape):
    model = AddTensorConvModule()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(
        model, input_shape, use_neutron_for_format_conversion=False
    )

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )


@pytest.mark.parametrize(
    "x_input_shape, y_input_shape",
    [
        pytest.param((1, 4, 7), (4, 7), id="3D -> 2D."),
        pytest.param((1, 4, 8), (1, 4, 4, 8), id="3D -> 4D."),
        pytest.param((1, 1, 4, 4, 8), (1, 4, 4, 8), id="5D -> 4D."),
        pytest.param((4,), (4, 4), id="1D -> 2D."),
        pytest.param((4,), (4, 4, 4), id="1D -> 3D."),
        pytest.param((6, 6), (1, 8, 6, 6), id="2D -> 4D."),
        pytest.param((6, 6), (6,), id="2D -> 1D."),
    ],
)
def test_add_tensor_broadcasting_unsupported_quant_conversion(
    x_input_shape, y_input_shape
):
    model = AddTensorModule()

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, [x_input_shape, y_input_shape]
    ).exported_program()
    nodes = list(edge_program.graph.nodes)

    # Broadcast is not supported, node is not converted
    assert nodes[6].target.__name__ == "aten.add.Tensor"  # Add Tensor is not delegated.

    # Capture converted program
    # exported_program: ExportedProgram = converter_spy.call_args.args[1]
    #
    # x_input_data = (np.random.random(x_input_shape).astype(np.float32) * 50).astype(np.int8)
    # y_input_data = (np.random.random(y_input_shape).astype(np.float32) * 50).astype(np.int8)
    # input_data = {0: x_input_data, 1: y_input_data}
    #
    # convert_run_compare(exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data)
