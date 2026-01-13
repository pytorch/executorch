# Copyright 2024-2025 NXP
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
from executorch.backends.nxp.backend.ir.conversion_config import ConversionConfig
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
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch import nn
from torch.export import ExportedProgram
from executorch.backends.nxp.tests.use_qat import *  # noqa F403


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


class ConvViewLinearModule(torch.nn.Module):
    def __init__(self, view_new_shape: list[int], channels: int, bias: bool):
        super().__init__()
        self.view_new_shape = view_new_shape
        self.conv = nn.Conv2d(channels, channels, 1, 1)
        self.linear = nn.Linear(view_new_shape[1], 8, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(self.view_new_shape)
        x = self.linear(x)
        return x


class ConvViewConvModule(torch.nn.Module):
    def __init__(self, view_new_shape: list[int], channels: int):
        super().__init__()
        self.view_new_shape = view_new_shape
        self.conv1 = nn.Conv2d(channels, channels, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(self.view_new_shape)
        x = self.conv2(x)
        return x


def test__view_copy__channels_first_to_2d(mocker):
    input_shape = (1, 4, 7, 9)
    new_shape = (6, 32)  # Mix up the dimensions for a thorough test.

    torch_model = ConvReshapeModule(channels=input_shape[1], new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(
        edge_program, input_data, tflite_input_preprocess=ToChannelLastPreprocess()
    )

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Conv2D)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Reshape)


def test__view_copy__channels_first_to_4d(mocker):
    input_shape = (1, 8, 6, 8)
    new_shape = (7, 4, 2, 5)

    torch_model = ConvReshapeModule(channels=input_shape[1], new_shape=new_shape)
    edge_program = to_edge_program(torch_model, input_shape).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    converter_spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        atol=2.0e-7,
        conversion_config=ConversionConfig(
            {"use_neutron_for_format_conversion": False}
        ),
    )

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Conv2D)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Reshape)


def test__view_copy__formatless_to_channels_first(mocker):
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
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        atol=2.0e-7,
    )

    tflite_model = converter_spy.spy_return
    ops = tflite_model.sub_graphs[0].operators.vector
    assert len(ops) == 3
    assert isinstance(ops[0].builtin_options, Reshape)
    assert isinstance(ops[1].builtin_options, Transpose)
    assert isinstance(ops[2].builtin_options, Conv2D)


def test__view_copy__formatless_to_formatless(mocker):
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
def test_view_copy_w_linear_quant_conversion(mocker, input_shape, new_shape, use_qat):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(
        LinearReshapeModule(new_shape=new_shape), input_shape, use_qat=use_qat
    )

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
def test_view_w_conv_linear_quant_conversion(
    mocker, input_shape, channels_view_out, use_qat
):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    # Run conversion
    _ = to_quantized_edge_program(
        ConvLinearViewModule(
            channels=input_shape[1], channels_view_out=channels_view_out
        ),
        input_shape,
        use_qat=use_qat,
        use_neutron_for_format_conversion=False,
    )

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    # Capture converted program
    edge_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        atol=1.0,
    )


@pytest.mark.parametrize(
    "bias",
    [True, False],
)
def test__view_copy__context_dependent__channels_first_to_formatless__transpose_fused(
    bias, mocker
):
    input_shape = (1, 2, 3, 4)
    new_shape = [1, 2 * 3 * 4]
    module = ConvViewLinearModule(new_shape, 2, bias)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    ep = to_quantized_edge_program(
        module,
        input_shape,
        use_neutron_for_format_conversion=False,
    ).exported_program()

    # Make sure all 3 nodes were delegated
    assert any(n.name == "executorch_call_delegate" for n in ep.graph.nodes)
    assert not graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.addmm.default,
            exir_ops.edge.aten.view_copy.default,
        ],
    )

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    converted_edge_program = converter_spy.call_args.args[1]
    neutron_ir_model = converter_spy.spy_return[0]
    convert_run_compare(
        converted_edge_program,
        input_data,
        tfl_model=neutron_ir_model,
        tflite_input_preprocess=ToChannelLastPreprocess(),
    )


@pytest.mark.parametrize(
    "bias",
    [True, False],
)
def test__view_copy__context_dependent__channels_first_to_formatless__transpose_not_fusable(
    bias,
):
    input_shape = (1, 2, 3, 4)
    new_shape = [
        2,
        3 * 4,
    ]  # The batch size changes, which makes the optimization not applicable.
    module = ConvViewLinearModule(new_shape, 2, bias)

    ep = to_quantized_edge_program(
        module,
        input_shape,
        use_neutron_for_format_conversion=False,
    ).exported_program()

    # Make sure the convolution and the linear were delegated, but not the view_copy.
    assert any(n.name == "executorch_call_delegate" for n in ep.graph.nodes)
    assert not graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.addmm.default,
        ],
    )
    assert graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.view_copy.default,
        ],
    )


def test__view_copy__formatless_to_channels_first__transpose_supported(mocker):
    input_shape = (1, 8 * 3 * 8)
    new_shape = [1, 8, 3, 8]
    module = FormatlessToChannelsFirstModule(8, new_shape)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    ep = to_quantized_edge_program(
        module,
        input_shape,
        use_neutron_for_format_conversion=False,
    ).exported_program()

    # Make sure both nodes were delegated
    assert any(n.name == "executorch_call_delegate" for n in ep.graph.nodes)
    assert not graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.view_copy.default,
        ],
    )

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    converted_edge_program = converter_spy.call_args.args[1]
    neutron_ir_model = converter_spy.spy_return[0]
    convert_run_compare(
        converted_edge_program,
        input_data,
        tfl_model=neutron_ir_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )


def test__view_copy__formatless_to_channels_first__transpose_not_supported():
    input_shape = (1, 8 * 3 * 4)
    new_shape = [1, 8, 3, 4]  # The last dim is not a multiple of num_macs.
    module = FormatlessToChannelsFirstModule(8, new_shape)

    ep = to_quantized_edge_program(
        module,
        input_shape,
        use_neutron_for_format_conversion=False,
    ).exported_program()

    # Make sure the view_copy was not delegated.
    assert any(n.name == "executorch_call_delegate" for n in ep.graph.nodes)
    assert not graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.convolution.default,
        ],
    )
    assert graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.view_copy.default,
        ],
    )


def test__view_copy__channels_first_to_channels_first__transpose_supported(mocker):
    input_shape = (1, 8, 3, 8)
    new_shape = [1, 8, 1, 24]
    module = ConvViewConvModule(new_shape, 8)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    ep = to_quantized_edge_program(
        module,
        input_shape,
        use_neutron_for_format_conversion=False,
    ).exported_program()

    # Make sure all nodes were delegated
    assert any(n.name == "executorch_call_delegate" for n in ep.graph.nodes)
    assert not graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.convolution.default,
            exir_ops.edge.aten.view_copy.default,
        ],
    )

    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)

    converted_edge_program = converter_spy.call_args.args[1]
    neutron_ir_model = converter_spy.spy_return[0]
    convert_run_compare(
        converted_edge_program,
        input_data,
        tfl_model=neutron_ir_model,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
    )


def test__view_copy__channels_first_to_channels_first__transpose_not_supported():
    input_shape = (1, 8, 3, 5)  # The last dimension is not a multiple of num_macs.
    new_shape = [1, 8, 1, 15]
    module = ConvViewConvModule(new_shape, 8)

    ep = to_quantized_edge_program(
        module,
        input_shape,
        use_neutron_for_format_conversion=False,
    ).exported_program()

    # Make sure the view_copy was NOT delegated
    assert any(n.name == "executorch_call_delegate" for n in ep.graph.nodes)
    assert not graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.convolution.default,
        ],
    )
    assert graph_contains_any_of_ops(
        ep.graph,
        [
            exir_ops.edge.aten.view_copy.default,
        ],
    )
