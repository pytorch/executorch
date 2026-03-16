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
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import MeanDimConvModule, MeanDimLinearModule
from executorch.backends.nxp.tests.use_qat import *  # noqa F403
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class MeanDimModule(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 4, 8, 8), (-1, -2), id="Dim -1, -2."),
        pytest.param((1, 4, 8, 8), (-2, -1), id="Dim -2, -1."),
        pytest.param((1, 4, 8, 8), (2, 3), id="Dim 2, 3."),
        pytest.param((1, 4, 8, 8), (3, 2), id="Dim 3, 2."),
    ],
)
def test_mean_dim_conv_quant_conversion(
    mocker, input_shape, dim, use_qat, keepdim=True
):
    model = MeanDimConvModule(dim, keepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()
    # Make sure the `mean.dim` was delegated.
    assert not graph_contains_any_of_ops(ep.graph, [exir_ops.edge.aten.mean.dim])
    assert any("lowered_module" in n.name for n in ep.graph.nodes)

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
        atol=1.0,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 32), 0, id="Dim 0."),
        pytest.param((1, 32), 1, id="Dim 1."),
    ],
)
@pytest.mark.parametrize(
    "keepdim",
    [
        pytest.param(False, id="Don't keep dim."),
        pytest.param(True, id="Keep dim."),
    ],
)
def test_mean_dim_linear_unsupported_quant_conversion(
    mocker, input_shape, dim, use_qat, keepdim
):
    model = MeanDimLinearModule(dim, keepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()
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
    "keepdim",
    [
        pytest.param(False, id="Don't keep dim."),
        pytest.param(True, id="Keep dim."),
    ],
)
def test_mean_dim_conv_unsupported_quant_conversion(
    mocker, input_shape, dim, use_qat, keepdim
):
    model = MeanDimConvModule(dim, keepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    edge_program = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat, use_neutron_for_format_conversion=False
    ).exported_program()
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


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 2, 3, 8), (1, 2), id="Dim 1, 2."),
        pytest.param((1, 2, 3, 8), (2, 1), id="Dim 2, 1."),
        pytest.param((1, 2, 3, 8), (-3, -2), id="Dim -3, -2."),
        pytest.param((1, 2, 3, 8), (-2, -3), id="Dim -2, -3."),
    ],
)
def test_mean_dim__formatless__supported(
    mocker, input_shape, dim, use_qat, keepdim=True
):
    model = MeanDimModule(dim, keepdim)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `mean.dim` was delegated.
    assert not graph_contains_any_of_ops(ep.graph, [exir_ops.edge.aten.mean.dim])
    assert any("lowered_module" in n.name for n in ep.graph.nodes)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        exported_program,
        input_data=input_data,
        tfl_model=tflite_flatbuffers_model,
        atol=1,
    )


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param((1, 2, 3, 8), (2, 3), id="Dim 2, 3."),
    ],
)
def test_mean_dim__formatless__unsupported(input_shape, dim, use_qat, keepdim=True):
    model = MeanDimModule(dim, keepdim)

    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `mean.dim` was NOT delegated.
    assert graph_contains_any_of_ops(ep.graph, [exir_ops.edge.aten.mean.dim])
    assert not any("lowered_module" in n.name for n in ep.graph.nodes)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param(
            (1, 8, 8, 4), (1, 2), id="Dim 1, 2 (supported), channels = 4 (unsupported)."
        ),
    ],
)
def test_mean_dim__formatless__unsupported_channels(
    input_shape, dim, use_qat, keepdim=True
):
    model = MeanDimModule(dim, keepdim)

    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `mean.dim` was NOT delegated.
    assert graph_contains_any_of_ops(ep.graph, [exir_ops.edge.aten.mean.dim])
    assert not any("lowered_module" in n.name for n in ep.graph.nodes)


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        pytest.param(
            (1, 4, 8, 8), (2, 3), id="Dim 2, 3 (supported), channels = 5 (unsupported)."
        ),
    ],
)
def test_mean_dim__channels_first__unsupported_channels(
    input_shape, dim, use_qat, keepdim=True
):
    model = MeanDimConvModule(
        dim, keepdim, out_channels=5
    )  # Only multiples of 8 (num_macs) are supported.

    # Run conversion
    ep = to_quantized_edge_program(
        model, input_shape, use_qat=use_qat
    ).exported_program()

    # Make sure the `mean.dim` was NOT delegated.
    assert graph_contains_any_of_ops(ep.graph, [exir_ops.edge.aten.mean.dim])
