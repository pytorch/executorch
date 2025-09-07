# Copyright 2024-2025 NXP
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
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.backends.nxp.tests.models import Conv1dModule, Conv2dModule
from torch.export import ExportedProgram


@pytest.fixture(autouse=True)
def reseed_model_per_test_run():
    torch.manual_seed(23)
    np.random.seed(23)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [2, 1])
@pytest.mark.parametrize("kernel_size", [(1,), (3,)])
def test_conv1d_quant_conversion(stride, dilation, kernel_size, mocker):
    input_shape = (1, 4, 16)
    model = Conv1dModule(stride=stride, dilation=dilation, kernel_size=kernel_size)
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    ops_spy = mocker.spy(ModelBuilder, "finish")

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
        atol=1.0,
    )

    # Capture IR model ops
    conversion_result = ops_spy.spy_return
    ops = conversion_result.sub_graphs[0].operators.vector

    assert len(ops) == 3
    assert ops[0].builtin_options.operator_type == BuiltinOperator.RESHAPE
    assert ops[1].builtin_options.operator_type == BuiltinOperator.CONV_2D
    assert ops[2].builtin_options.operator_type == BuiltinOperator.RESHAPE


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [2, 1])
@pytest.mark.parametrize("kernel_size", [(1,), (3,)])
@pytest.mark.parametrize("padding", [(1,), 2])
def test_conv1d_quant_conversion__padded(
    stride, dilation, kernel_size, padding, mocker
):
    input_shape = (1, 4, 16)
    model = Conv1dModule(
        stride=stride, dilation=dilation, kernel_size=kernel_size, padding=padding
    )
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    ops_spy = mocker.spy(ModelBuilder, "finish")

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
        atol=1.0,
    )

    # Capture IR model ops
    conversion_result = ops_spy.spy_return
    ops = conversion_result.sub_graphs[0].operators.vector

    assert len(ops) == 4
    assert ops[0].builtin_options.operator_type == BuiltinOperator.RESHAPE
    assert ops[1].builtin_options.operator_type == BuiltinOperator.PADV2
    assert ops[2].builtin_options.operator_type == BuiltinOperator.CONV_2D
    assert ops[3].builtin_options.operator_type == BuiltinOperator.RESHAPE

    # Make sure the padding used the `zero-point`.
    pad_value = ops[1].tmp_inputs[2].tmp_buffer.data.item()
    assert (
        pad_value == ops[1].tmp_inputs[0].quantization.zero_point[0]
    )  # `Pad` input zp.
    assert (
        pad_value == ops[1].tmp_outputs[0].quantization.zero_point[0]
    )  # `Pad` output zp.
    assert (
        pad_value == ops[2].tmp_inputs[0].quantization.zero_point[0]
    )  # `Conv` input zp.


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [2, 1])
@pytest.mark.parametrize("kernel_size", [(1,), (3,)])
def test_conv1d_quant_conversion__depthwise(stride, dilation, kernel_size, mocker):
    input_shape = (1, 4, 16)
    group = input_shape[1]
    model = Conv1dModule(
        group=group,
        in_channels=group,
        out_channels=group,
        stride=stride,
        dilation=dilation,
        kernel_size=kernel_size,
    )
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    ops_spy = mocker.spy(ModelBuilder, "finish")

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
        atol=1.0,
    )

    # Capture IR model ops
    ops = ops_spy.spy_return.sub_graphs[0].operators.vector

    assert len(ops) == 3
    assert ops[0].builtin_options.operator_type == BuiltinOperator.RESHAPE
    assert ops[1].builtin_options.operator_type == BuiltinOperator.DEPTHWISE_CONV_2D
    assert ops[2].builtin_options.operator_type == BuiltinOperator.RESHAPE


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [2, 1])
@pytest.mark.parametrize("kernel_size", [(1,), (3,)])
@pytest.mark.parametrize("padding", [(1,), 2])
def test_conv1d_quant_conversion__depthwise__padded(
    stride, dilation, kernel_size, padding, mocker
):
    input_shape = (1, 4, 16)
    group = input_shape[1]
    model = Conv1dModule(
        group=group,
        in_channels=group,
        out_channels=group,
        stride=stride,
        dilation=dilation,
        kernel_size=kernel_size,
        padding=padding,
    )
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    ops_spy = mocker.spy(ModelBuilder, "finish")

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
        atol=1.0,
    )

    # Capture IR model ops
    ops = ops_spy.spy_return.sub_graphs[0].operators.vector

    assert len(ops) == 4
    assert ops[0].builtin_options.operator_type == BuiltinOperator.RESHAPE
    assert ops[1].builtin_options.operator_type == BuiltinOperator.PADV2
    assert ops[2].builtin_options.operator_type == BuiltinOperator.DEPTHWISE_CONV_2D
    assert ops[3].builtin_options.operator_type == BuiltinOperator.RESHAPE

    # Make sure the padding used the `zero-point`.
    pad_value = ops[1].tmp_inputs[2].tmp_buffer.data.item()
    assert (
        pad_value == ops[1].tmp_inputs[0].quantization.zero_point[0]
    )  # `Pad` input zp.
    assert (
        pad_value == ops[1].tmp_outputs[0].quantization.zero_point[0]
    )  # `Pad` output zp.
    assert (
        pad_value == ops[2].tmp_inputs[0].quantization.zero_point[0]
    )  # `Conv` input zp.


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
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tfl_model=tflite_flatbuffers_model,
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        input_data=input_data,
        atol=1.0,
    )


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("kernel_shape", [[1, 2], [3, 3], [4, 1]])
def test_conv2d_conversion__depthwise(stride, dilation, kernel_shape, mocker):
    input_shape = (1, 3, 12, 16)
    group = input_shape[1]
    edge_program = to_edge_program(
        Conv2dModule(
            group=group,
            in_channels=group,
            out_channels=group,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_shape,
        ),
        input_shape,
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        atol=4e-7,
    )
    conversion_result = spy.spy_return
    ops = conversion_result.sub_graphs[0].operators.vector

    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.DEPTHWISE_CONV_2D


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("kernel_shape", [[1, 2], [3, 3], [4, 1]])
def test_conv2d_conversion__depthwise__quantized(
    stride, dilation, kernel_shape, mocker
):
    input_shape = (1, 4, 12, 12)
    group = input_shape[1]
    spy = mocker.spy(ModelBuilder, "finish")

    edge_program = to_quantized_edge_program(
        Conv2dModule(
            group=group,
            in_channels=group,
            out_channels=group,
            stride=stride,
            dilation=dilation,
            kernel_size=kernel_shape,
        ),
        tuple(input_shape),
    ).exported_program()

    ops = spy.spy_return.sub_graphs[0].operators.vector
    assert len(ops) == 1
    assert ops[0].builtin_options.operator_type == BuiltinOperator.DEPTHWISE_CONV_2D

    nodes = list(edge_program.graph.nodes)
    assert (
        len(nodes) == 7
    )  # input, Quant, lowered_module, delegate_call, getitem, Deq, output
    assert nodes[2].target == "lowered_module_0"


@pytest.mark.parametrize("padding", [1, 2])
def test_conv2d_conversion__depthwise__padded(padding, mocker):
    input_shape = (1, 3, 13, 15)
    group = input_shape[1]
    edge_program = to_edge_program(
        Conv2dModule(
            group=group, in_channels=group, out_channels=group, padding=padding
        ),
        input_shape,
    ).exported_program()

    input_data = np.random.random(input_shape).astype(np.float32)

    spy = mocker.spy(ModelBuilder, "finish")

    convert_run_compare(
        edge_program,
        input_data,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        atol=4e-7,
    )
    conversion_result = spy.spy_return
    ops = conversion_result.sub_graphs[0].operators.vector

    assert len(ops) == 2
    assert ops[0].builtin_options.operator_type == BuiltinOperator.PAD
    assert ops[1].builtin_options.operator_type == BuiltinOperator.DEPTHWISE_CONV_2D


@pytest.mark.parametrize("padding", [1, 2])
def test_conv2d_conversion__depthwise__padded__quantized(padding, mocker):
    input_shape = (1, 4, 12, 12)
    group = input_shape[1]
    spy = mocker.spy(ModelBuilder, "finish")

    edge_program = to_quantized_edge_program(
        Conv2dModule(
            group=group, in_channels=group, out_channels=group, padding=padding
        ),
        tuple(input_shape),
    ).exported_program()

    ops = spy.spy_return.sub_graphs[0].operators.vector
    assert len(ops) == 2
    assert ops[0].builtin_options.operator_type == BuiltinOperator.PADV2
    assert ops[1].builtin_options.operator_type == BuiltinOperator.DEPTHWISE_CONV_2D

    nodes = list(edge_program.graph.nodes)
    assert (
        len(nodes) == 7
    )  # input, Quant, lowered_module, delegate_call, getitem, Deq, output
    assert nodes[2].target == "lowered_module_0"
