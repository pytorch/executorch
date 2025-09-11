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
from executorch.backends.nxp.tests.models import MeanDimConvModule, MeanDimLinearModule
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 4, 8, 8), (-1, -2), id="Dim -1, -2."),
    ],
)
def test_mean_dim_conv_quant_conversion(mocker, input_shape, dim, keeepdim=True):
    model = MeanDimConvModule(dim, keeepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(model, input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        input_data=input_data,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        tfl_model=tflite_flatbuffers_model,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 32), 0, id="Dim 0."),
        pytest.param((1, 32), 1, id="Dim 1."),
    ],
)
@pytest.mark.parametrize(
    "keeepdim",
    [
        pytest.param(False, id="Don't keep dim."),
        pytest.param(True, id="Keep dim."),
    ],
)
def test_mean_dim_linear_unsupported_quant_conversion(
    mocker, input_shape, dim, keeepdim
):
    model = MeanDimLinearModule(dim, keeepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(model, input_shape).exported_program()
    nodes = list(edge_program.graph.nodes)

    # Last 2 dimensions are not used or keepdim is False, cannot be converted to MeanDim, node is not delegated
    assert nodes[6].target.__name__ == "aten.mean.dim"

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 4, 8, 8), 0, id="Dim 0."),
        pytest.param((1, 4, 8, 8), 2, id="Dim 2."),
        pytest.param((1, 4, 8, 8), -1, id="Dim -1."),
        pytest.param((1, 4, 8, 8), -2, id="Dim -2."),
        pytest.param((1, 4, 8, 8), (0, 1), id="Dim 0, 1."),
        pytest.param((1, 4, 8, 8), (1, 3), id="Dim 1, 3."),
        pytest.param((1, 4, 8, 8), (-1, -3), id="Dim -1, -3."),
    ],
)
@pytest.mark.parametrize(
    "keeepdim",
    [
        pytest.param(False, id="Don't keep dim."),
        pytest.param(True, id="Keep dim."),
    ],
)
def test_mean_dim_conv_unsupported_quant_conversion(mocker, input_shape, dim, keeepdim):
    model = MeanDimConvModule(dim, keeepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(model, input_shape).exported_program()
    nodes = list(edge_program.graph.nodes)

    # Last 2 dimensions are not used or keepdim is False, cannot be converted to MeanDim, node is not delegated
    assert nodes[6].target.__name__ == "aten.mean.dim"

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        input_data=input_data,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        tfl_model=tflite_flatbuffers_model,
    )
