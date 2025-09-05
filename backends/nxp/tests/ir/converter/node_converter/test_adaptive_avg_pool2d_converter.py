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
    AdaptiveAvgPool2dConvMeanDimModule,
    AdaptiveAvgPool2dConvModule,
)
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape, output_size",
    [
        pytest.param(
            (1, 4, 16, 16), (4, 4), id="Pooling with equal height and width kernel."
        ),
        pytest.param(
            (1, 4, 16, 16), (8, 8), id="Pooling with equal height and width kernel."
        ),
        pytest.param((1, 4, 16, 16), (4, 8), id="Pooling with height > width kernel."),
        pytest.param((1, 4, 16, 22), (4, 11), id="Pooling with height > width kernel."),
        pytest.param((1, 4, 32, 32), (16, 4), id="Pooling with height < width kernel."),
        pytest.param((1, 4, 32, 16), (16, 4), id="Pooling with height < width kernel."),
    ],
)
def test_adaptive_avg_pool_2d_delegated_quant_conversion(
    mocker, input_shape, output_size
):
    model = AdaptiveAvgPool2dConvModule(output_size)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(model, input_shape).exported_program()
    nodes = [str(node) for node in edge_program.graph.nodes]

    # Input size is a multiple of output size, can be converted to AveragePool, node is delegated
    assert "aten__adaptive_avg_pool2d_default" not in nodes

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
        atol=1,
    )


@pytest.mark.parametrize(
    "input_shape, output_size",
    [
        pytest.param(
            (1, 4, 16, 16), (6, 6), id="Pooling with equal height and width kernel."
        ),
        pytest.param((1, 4, 16, 16), (4, 7), id="Pooling with height > width kernel."),
        pytest.param((1, 4, 16, 22), (4, 10), id="Pooling with height > width kernel."),
        pytest.param((1, 4, 32, 32), (14, 7), id="Pooling with height < width kernel."),
        pytest.param((1, 4, 32, 16), (15, 5), id="Pooling with height < width kernel."),
    ],
)
def test_adaptive_avg_pool_2d_non_delegated_quant_conversion(
    mocker, input_shape, output_size
):
    model = AdaptiveAvgPool2dConvModule(output_size)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(model, input_shape).exported_program()
    nodes = list(edge_program.graph.nodes)

    # Input size is not a multiple of output size, cannot be converted to AveragePool, node is not delegated
    assert str(nodes[6]) == "aten__adaptive_avg_pool2d_default"

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
        atol=1,
    )


def test_adaptive_avg_pool_2d_mean_dim_quant_conversion(mocker):
    input_shape = (1, 4, 16, 16)
    model = AdaptiveAvgPool2dConvMeanDimModule()

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
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
    )
