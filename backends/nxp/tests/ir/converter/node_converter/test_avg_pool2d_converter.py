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
from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOperator import (
    BuiltinOperator,
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
from executorch.backends.nxp.tests.models import AvgPool2dConvModule, AvgPool2dModule
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize(
    "input_shape, padding, count_include_pad",
    [
        pytest.param(
            (1, 4, 8, 8),
            (0, 0),
            True,
            id="No padding, include padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 8, 8),
            (0, 0),
            False,
            id="No padding, don't include padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 8, 8),
            (1, 1),
            True,
            id="Padding, keep the same output tensor size as input, include "
            "padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 8, 8),
            (1, 0),
            True,
            id="Padding, change the output tensor size, include padding to "
            "average calculation.",
        ),
        pytest.param(
            (1, 4, 9, 9),
            (1, 0),
            True,
            id="Padding, change the output tensor size, include padding to "
            "average calculation.",
        ),
        pytest.param(
            (1, 4, 7, 7),
            (0, 1),
            True,
            id="Padding, change the output tensor size, include padding to "
            "average calculation.",
        ),
    ],
)
def test_avg_pool_2d_conversion(input_shape, padding, count_include_pad):
    model = AvgPool2dModule(padding=padding, count_include_pad=count_include_pad)
    edge_program = to_edge_program(model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tflite_output_preprocess=ToNCHWPreprocess(),
    )


@pytest.mark.parametrize(
    "input_shape, padding, count_include_pad",
    [
        pytest.param(
            (1, 4, 16, 16),
            (0, 0),
            True,
            id="No padding, include padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 16, 16),
            (0, 0),
            False,
            id="No padding, don't include padding to average calculation.",
        ),
        pytest.param(
            (1, 4, 16, 16),
            (1, 1),
            True,
            id="Keep the same output tensor size as input, include padding "
            "to average calculation.",
        ),
        pytest.param(
            (1, 4, 16, 16),
            (1, 0),
            True,
            id="Padding, change same tensor size, include padding to average"
            " calculation.",
        ),
        pytest.param(
            (1, 4, 11, 11),
            (0, 1),
            True,
            id="Padding, change same tensor size, include padding to average"
            " calculation.",
        ),
        pytest.param(
            (1, 4, 11, 11),
            (1, 0),
            True,
            id="Padding, change same tensor size, include padding to average"
            " calculation.",
        ),
    ],
)
def test_avg_pool_2d_quant_conversion(mocker, input_shape, padding, count_include_pad):
    model = AvgPool2dConvModule(padding=padding, count_include_pad=count_include_pad)

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
    )


def test_avg_pool_2d_quant_conversion__padded(mocker):
    input_shape = (1, 8, 8, 8)
    model = AvgPool2dModule(True, 1)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    ops_spy = mocker.spy(ModelBuilder, "finish")

    # Run conversion
    _ = to_quantized_edge_program(model, input_shape)

    # Capture the converter operators.
    ops = ops_spy.spy_return.sub_graphs[0].operators.vector

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
    )

    assert len(ops) == 2
    assert ops[0].builtin_options.operator_type == BuiltinOperator.PADV2
    assert ops[1].builtin_options.operator_type == BuiltinOperator.AVERAGE_POOL_2D

    # Make sure the padding used the `zero-point`.
    pad_value = ops[0].tmp_inputs[2].tmp_buffer.data.item()
    assert (
        pad_value == ops[0].tmp_inputs[0].quantization.zero_point[0]
    )  # `Pad` input zp.
    assert (
        pad_value == ops[0].tmp_outputs[0].quantization.zero_point[0]
    )  # `Pad` output zp.
    assert (
        pad_value == ops[1].tmp_inputs[0].quantization.zero_point[0]
    )  # `AvgPool` input zp.
