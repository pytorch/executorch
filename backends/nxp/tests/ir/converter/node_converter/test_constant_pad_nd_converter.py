# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import (
    to_edge_program,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    ToNCHWPreprocess,
    ToNHWCPreprocess,
)
from executorch.backends.nxp.tests.models import (
    ConstantPadNDConvModule,
    ConstantPadNDModule,
    Conv2dConstantPadNDModule,
)
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("constant", [0.0, 42.0, -13.37])
def test_constant_pad_nd_conversion__specific_constant(constant):
    input_shape = [2, 4, 6, 8]
    paddings = [1, 2, 3, 4]

    edge_program = to_edge_program(
        ConstantPadNDModule(paddings, constant), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(edge_program, input_data)


@pytest.mark.parametrize("constant", [0.0, 67.28, 42.0, -13.37])
@pytest.mark.skip(reason="Neutron Converter does not fully convert for NPU")
def test_constant_pad_nd_quant_conversion__specific_constant(mocker, constant):
    input_shape = (2, 4, 12, 12)
    paddings = (2, 2, 2, 2)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(
        Conv2dConstantPadNDModule(paddings, constant), input_shape
    )

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    edge_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        edge_program,
        input_data,
        tfl_model=tflite_flatbuffers_model,
        atol=1.0,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
    )


def test_constant_pad_nd_conversion__default_constant():
    input_shape = [2, 4, 6, 8]
    paddings = [1, 2, 3, 4]

    edge_program = to_edge_program(
        ConstantPadNDModule(paddings), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(edge_program, input_data)


@pytest.mark.parametrize(
    "input_shape, paddings",
    [
        pytest.param([2], list(range(2)), id="1D, padding H"),
        pytest.param([2, 4], list(range(2)), id="2D, padding H"),
        pytest.param([2, 4], list(range(4)), id="2D, padding N, H"),
        pytest.param([2, 4, 6], list(range(2)), id="3D, padding H"),
        pytest.param([2, 4, 6], list(range(4)), id="3D, padding C, H"),
        pytest.param([2, 4, 6], list(range(6)), id="3D, padding N, C, H"),
        pytest.param([2, 4, 6, 8], list(range(2)), id="4D, padding W"),
        pytest.param([2, 4, 6, 8], list(range(4)), id="4D, padding H, W"),
        pytest.param([2, 4, 6, 8], list(range(6)), id="4D, padding C, H, W"),
        pytest.param([2, 4, 6, 8], list(range(8)), id="4D, padding N, C, H, W"),
        pytest.param([1, 2, 3, 4, 5], list(range(2)), id="5D, padding D"),
        pytest.param([1, 2, 3, 4, 5], list(range(4)), id="5D, padding W, D"),
        pytest.param([1, 2, 3, 4, 5], list(range(6)), id="5D, padding H, W, D"),
        pytest.param([1, 2, 3, 4, 5], list(range(8)), id="5D, padding C, H, W, D"),
        pytest.param([1, 2, 3, 4, 5], list(range(10)), id="5D, padding N, C, H, W, D"),
    ],
)
def test_constant_pad_nd_conversion__format_less(input_shape, paddings):
    edge_program = to_edge_program(
        ConstantPadNDModule(paddings), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(edge_program, input_data)


@pytest.mark.parametrize(
    "input_shape, paddings",
    [
        pytest.param([2, 4, 6, 8], list(range(2)), id="4D, padding W"),
        pytest.param([2, 4, 6, 8], list(range(4)), id="4D, padding H, W"),
        pytest.param([2, 1, 6, 8], [1, 2, 3, 4, 2, 1], id="4D, padding C, H, W"),
        pytest.param(
            [2, 1, 6, 8], [1, 2, 3, 4, 2, 1, 5, 6], id="4D, padding N, C, H, W"
        ),
    ],
)
def test_constant_pad_nd_conversion__channels_first(input_shape, paddings):
    edge_program = to_edge_program(
        ConstantPadNDConvModule(paddings), input_shape
    ).exported_program()  # Extra `Conv` after the padding.

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
    )
