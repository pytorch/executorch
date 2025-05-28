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
from executorch.backends.nxp.tests.models import Conv2dModule
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape, padding",
    [
        pytest.param((1, 4, 32, 32), (0, 0), id="No padding."),
        pytest.param(
            (1, 4, 32, 32),
            (1, 1),
            id="Padding, keep the same output tensor size as input.",
        ),
        pytest.param(
            (1, 4, 32, 32), (1, 0), id="Padding, change the output tensor size."
        ),
        pytest.param(
            (1, 4, 31, 31), (1, 0), id="Padding, change the output tensor size."
        ),
        pytest.param(
            (1, 4, 31, 31), (0, 1), id="Padding, change the output tensor size."
        ),
    ],
)
@pytest.mark.parametrize(
    "dilation",
    [
        pytest.param(1, id="No dilation."),
        pytest.param(2, id="2 dilation."),
        pytest.param((1, 3), id="Side-different dilation."),
    ],
)
def test_conv2d_conversion(input_shape, padding, dilation: int):
    edge_program = to_edge_program(
        Conv2dModule(padding=padding, dilation=dilation), input_shape
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
        atol=4e-7,
    )


@pytest.mark.parametrize(
    "model, input_shape",
    [
        pytest.param(
            Conv2dModule(in_channels=8, out_channels=32, kernel_size=5),
            (1, 8, 32, 32),
            id="In ch 8, out ch 32, kernel 5",
        ),
        pytest.param(
            Conv2dModule(in_channels=8, out_channels=32, kernel_size=5, padding=3),
            (1, 8, 32, 32),
            id="In ch 8, out ch 32, kernel 5, padding 3",
        ),
        pytest.param(
            Conv2dModule(in_channels=8, out_channels=32, kernel_size=5, padding=(2, 3)),
            (1, 8, 31, 31),
            id="In ch 8, out ch 32, kernel 5, padding (2, 3)",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=8,
                out_channels=32,
                kernel_size=5,
                padding=(2, 3),
                dilation=(1, 2),
            ),
            (1, 8, 31, 31),
            id="In ch 8, out ch 32, kernel 5, padding (2, 3), dilation (1, 2)",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=16, out_channels=32, kernel_size=3, padding=2, dilation=2
            ),
            (1, 16, 32, 32),
            id="In ch 16, out ch 32, kernel 3, padding 2, dilation 2",
        ),
        pytest.param(
            Conv2dModule(in_channels=32, out_channels=32, kernel_size=3, dilation=2),
            (1, 32, 32, 32),
            id="In ch 32, out ch 32, kernel 3, dilation 2",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=(0, 1),
                dilation=2,
            ),
            (1, 32, 35, 35),
            id="In ch 32, out ch 32, kernel 3, padding (0, 1), dilation 2",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=(1, 0),
                dilation=(3, 1),
            ),
            (1, 32, 35, 35),
            id="In ch 32, out ch 32, kernel 3, padding (1, 0), dilation (3, 1)",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=32, out_channels=32, kernel_size=3, dilation=(2, 3)
            ),
            (1, 32, 32, 32),
            id="In ch 32, out ch 32, kernel 3, dilation (2, 3)",
        ),
        pytest.param(
            Conv2dModule(in_channels=32, out_channels=64, kernel_size=4),
            (1, 32, 32, 32),
            id="In ch 32, out ch 32, kernel 4",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=32, out_channels=64, kernel_size=4, padding=(1, 2)
            ),
            (1, 32, 33, 33),
            id="In ch 32, out ch 32, kernel 4, padding (1, 2)",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=32, out_channels=64, kernel_size=4, padding=(1, 0)
            ),
            (1, 32, 33, 33),
            id="In ch 32, out ch 32, kernel 4, padding (1, 0)",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=32, out_channels=64, kernel_size=4, padding=(0, 2)
            ),
            (1, 32, 32, 32),
            id="In ch 32, out ch 32, kernel 4, padding (0, 2)",
        ),
        pytest.param(
            Conv2dModule(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                padding=(0, 2),
                dilation=(1, 2),
            ),
            (1, 32, 32, 32),
            id="In ch 32, out ch 32, kernel 4, padding (0, 2), dilation (1, 2)",
        ),
    ],
)
def test_conv2d_quant_conversion(mocker, model: torch.nn.Module, input_shape):
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
        tflite_input_preprocess=ToNHWCPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToNCHWPreprocess(),
        input_data=input_data,
        atol=1.0,
    )
