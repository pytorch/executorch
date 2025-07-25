# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.converter.builder.model_builder import (
    ModelBuilder,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.conv_2d_options import (
    Conv2D,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.reshape_options import (
    Reshape,
)
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options.transpose_options import (
    Transpose,
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
from torch import nn
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


class FormatlessToChannelsFirstModule(nn.Module):
    def __init__(self, channels: int, new_shape: Sequence[int]):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 2, bias=True)
        self.new_shape = new_shape

    def forward(self, x):
        x = torch.reshape(x, self.new_shape)
        x = self.conv(x)
        return x


class FormatlessToFormatlessModule(nn.Module):
    def __init__(self, new_shape: Sequence[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = torch.reshape(x, self.new_shape)
        return x


class ConvReshapeModule(nn.Module):
    def __init__(self, channels: int, new_shape: Sequence[int]):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 2, bias=True)
        self.new_shape = new_shape

    def forward(self, x):
        x = self.conv(x)
        x = torch.reshape(x, self.new_shape)
        return x


class LinearReshapeModule(torch.nn.Module):
    def __init__(self, new_shape: Sequence[int]):
        super().__init__()
        self.linear = nn.Linear(64, 32, bias=True)
        self.new_shape = new_shape

    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, self.new_shape)
        return x


class ConvLinearViewModule(torch.nn.Module):
    def __init__(self, channels: int, channels_view_out: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2)
        self.linear = nn.Linear(channels_view_out, 32, bias=True)
        self.channels_view_out = channels_view_out
        self.avg_pool = nn.AvgPool2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(-1, self.channels_view_out)
        x = self.linear(x)
        return x


def test__channels_first_to_2d(mocker):
    input_shape = (1, 4, 7, 9)
    new_shape = (6, 32)  # Mix up the dimensions for a thorough test.

    torch_model = ConvReshapeModule(channels=input_shape[1], new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(
        edge_program, input_data, tflite_input_preprocess=ToNHWCPreprocess()
    )

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Conv2D)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Reshape)


def test__channels_first_to_4d(mocker):
    input_shape = (1, 8, 6, 8)
    new_shape = (7, 4, 2, 5)

    torch_model = ConvReshapeModule(channels=input_shape[1], new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        atol=2.0e-7,
    )

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Conv2D)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Reshape)


def test__formatless_to_channels_first(mocker):
    input_shape = (12, 32)
    new_shape = (1, 4, 12, 8)  # Mix up the dimensions for a thorough test.

    torch_model = FormatlessToChannelsFirstModule(
        channels=new_shape[1], new_shape=new_shape
    )
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(
        edge_program,
        input_data,
        tflite_output_preprocess=ToNCHWPreprocess(),
        atol=2.0e-7,
    )

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Reshape)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Conv2D)


def test__formatless_to_formatless(mocker):
    input_shape = (12, 32)
    new_shape = (1, 4, 6, 16)

    torch_model = FormatlessToFormatlessModule(new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(edge_program, input_data)

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 1  # No extra Transpose ops.
    assert isinstance(ops[0].builtin_options, Reshape)


@pytest.mark.parametrize(
    "input_shape, new_shape",
    [
        pytest.param((8, 64), (1, 16, 4, 4), id="2D"),
    ],
)
def test_view_copy_w_linear_quant_conversion(mocker, input_shape, new_shape):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(LinearReshapeModule(new_shape=new_shape), input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    edge_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        edge_program, input_data, tfl_model=tflite_flatbuffers_model, atol=1.0
    )


@pytest.mark.parametrize(
    "input_shape, channels_view_out",
    [
        pytest.param((1, 4, 16, 16), 196, id="4D"),
    ],
)
def test_view_w_conv_linear_quant_conversion(mocker, input_shape, channels_view_out):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(
        ConvLinearViewModule(
            channels=input_shape[1], channels_view_out=channels_view_out
        ),
        input_shape,
    )

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    edge_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        atol=1.0,
    )
