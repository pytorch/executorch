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
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.backends.nxp.tests.models import ConvWithSigmoid
from torch import nn
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


def test_conv_sigmoid(mocker, input_shape: tuple[int] = (1, 3, 112, 112)):
    model = ConvWithSigmoid(conv_in_channels=input_shape[1])

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        tfl_model=tflite_flatbuffers_model,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
        atol=1.0,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((10,), id="Scalar"),
        pytest.param((10, 25), id="1D"),
        pytest.param((10, 25, 25), id="2D"),
        pytest.param((10, 3, 25, 25), id="3D"),
        pytest.param((10, 3, 25, 25, 25), id="4D"),
    ],
)
def test_sigmoid_only(mocker, input_shape):
    model = nn.Sigmoid()

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    to_quantized_edge_program(model, input_shape).exported_program()

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape) * 50).astype(np.int8)
    convert_run_compare(
        exported_program, tfl_model=tflite_flatbuffers_model, input_data=input_data
    )
