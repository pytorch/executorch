# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.Model import Model
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.nxp_backend import PayloadComposer
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    EdgeProgramExecutor,
    graph_contains_any_of_ops,
    TFLiteExecutor,
    ToNHWCPreprocess,
)
from executorch.backends.nxp.tests.models import Conv2dModule, ConvFCSoftmaxModule
from torch.export import ExportedProgram


def test_lowered_program_and_tflite_output_match__conv2d__no_bias(mocker):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    model = Conv2dModule(bias=False)
    input_shape = (1, 4, 32, 32)

    # Run conversion
    to_quantized_edge_program(model, input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    tflite_model = Model.GetRootAs(tflite_flatbuffers_model)
    sub_graph = tflite_model.Subgraphs(0)

    assert sub_graph.OperatorsLength() == 1
    assert sub_graph.Operators(0).BuiltinOptionsType() == BuiltinOptions.Conv2DOptions

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (
        (torch.randn(input_shape, dtype=torch.float32) * 50)
        .type(torch.int8)
        .detach()
        .numpy()
    )
    input_data_tflite = np.transpose(input_data, [0, 2, 3, 1])

    # Execute program and TFLite model
    program_executor = EdgeProgramExecutor(exported_program)
    tflite_executor = TFLiteExecutor(model_content=tflite_flatbuffers_model)

    output_edge = program_executor.inference(input_data)
    output_tflite = tflite_executor.inference(input_data_tflite)

    output_tflite = np.transpose(output_tflite, [0, 3, 1, 2])

    # Outputs difference is smaller than 1 (rounding error in quantization)
    assert np.max(np.abs(output_edge - output_tflite)) <= 1


def test_conv_fc__lowered_program_and_tflite_output_match(mocker):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    model = ConvFCSoftmaxModule()
    input_shape = (1, 4, 5, 5)

    # Run conversion
    _ = to_quantized_edge_program(model, input_shape)

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    # Capture generated model
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # No Transpose ops in produced TFLite model
    tflite_subgraph = Model.GetRootAs(tflite_flatbuffers_model).Subgraphs(0)

    assert tflite_subgraph.OperatorsLength() == 3
    assert (
        tflite_subgraph.Operators(0).BuiltinOptionsType()
        == BuiltinOptions.Conv2DOptions
    )
    assert (
        tflite_subgraph.Operators(1).BuiltinOptionsType()
        == BuiltinOptions.ReshapeOptions
    )
    assert (
        tflite_subgraph.Operators(2).BuiltinOptionsType()
        == BuiltinOptions.FullyConnectedOptions
    )

    # Verify outputs of program and TFLite model
    input_data = (
        (torch.randn(input_shape, dtype=torch.float32))
        .type(torch.int8)
        .detach()
        .numpy()
    )
    convert_run_compare(
        exported_program,
        input_data=input_data,
        tflite_input_preprocess=ToNHWCPreprocess(),
    )


def test_delegating_format_related_transpose_operators__unsupported_shapes(mocker):
    # This test focuses on the case when Neutron would not support the inserted Transpose operators, so they are not
    #  inserted, so the runtime will permute the data.

    # Make sure none of the dimensions are multiples of `num_macs` (8), for proper testing.
    model = Conv2dModule(in_channels=3, out_channels=3, padding=1, stride=1)
    input_shape = (1, 3, 3, 3)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    payload_header_spy = mocker.spy(PayloadComposer, "_create_payload_header")
    edge_program = to_quantized_edge_program(
        model,
        input_shape,
        use_neutron_for_format_conversion=True,  # Make sure the IR converter inserts the extra `Transpose` operators.
    ).exported_program()

    # Make sure the edge_program only contains the 1 delegate call.
    nodes = list(edge_program.graph.nodes)
    assert len(nodes) == 7
    assert "call_delegate" in nodes[3].name
    assert not graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.aten.convolution.default]
    )
    assert not graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.aten.permute_copy.default]
    )

    # Capture the converted IR model.
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Make sure the `Transpose` ops are NOT in the IR model.
    tflite_subgraph = Model.GetRootAs(tflite_flatbuffers_model).Subgraphs(0)
    assert tflite_subgraph.OperatorsLength() == 2
    assert (
        tflite_subgraph.Operators(0).BuiltinOptionsType() == BuiltinOptions.PadV2Options
    )
    assert (
        tflite_subgraph.Operators(1).BuiltinOptionsType()
        == BuiltinOptions.Conv2DOptions
    )

    # Get the header of the payload for the delegated partition.
    payload_header = payload_header_spy.spy_return
    assert payload_header.size == 7
    # the 4th and 5th bytes indicate the format. `1` means `channels_last`, which means the runtime will transpose the data.
    assert all(payload_header[3:5] == [1, 1])  # [<input_byte>, <output_byte>]


def test_delegating_format_related_transpose_operators__supported_case(mocker):
    # Make sure the output channels (channels for the trailing Transpose), and the last input dimension (channels for
    #  the leading Transpose) are multiples of `num_macs``.

    num_macs = NeutronTargetSpec("imxrt700", "SDK_25_09").get_num_macs()
    model = Conv2dModule(
        in_channels=num_macs, out_channels=num_macs, padding=1, stride=1
    )
    input_shape = (1, num_macs, num_macs, num_macs)

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    payload_header_spy = mocker.spy(PayloadComposer, "_create_payload_header")
    edge_program = to_quantized_edge_program(
        model,
        input_shape,
        use_neutron_for_format_conversion=True,  # Make sure the IR converter inserts the extra `Transpose` operators.
    ).exported_program()

    # Make sure the edge_program only contains the 1 delegate call.
    nodes = list(edge_program.graph.nodes)
    assert len(nodes) == 7
    assert "call_delegate" in nodes[3].name
    assert not graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.aten.convolution.default]
    )
    assert not graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.aten.permute_copy.default]
    )

    # Capture the converted IR model.
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Make sure the `Transpose` ops ARE in the IR model.
    tflite_subgraph = Model.GetRootAs(tflite_flatbuffers_model).Subgraphs(0)
    assert tflite_subgraph.OperatorsLength() == 4
    assert (
        tflite_subgraph.Operators(0).BuiltinOptionsType()
        == BuiltinOptions.TransposeOptions
    )
    assert (
        tflite_subgraph.Operators(1).BuiltinOptionsType() == BuiltinOptions.PadV2Options
    )
    assert (
        tflite_subgraph.Operators(2).BuiltinOptionsType()
        == BuiltinOptions.Conv2DOptions
    )
    assert (
        tflite_subgraph.Operators(3).BuiltinOptionsType()
        == BuiltinOptions.TransposeOptions
    )

    # Get the header of the payload for the delegated partition.
    payload_header = payload_header_spy.spy_return
    assert payload_header.size == 7
    # the 4th and 5th bytes indicate the format. `0` means `channels_last`, which means the runtime will NOT transpose the data.
    assert all(payload_header[3:5] == [0, 0])  # [<input_byte>, <output_byte>]


def test_delegating_format_related_transpose_operators__supported_output__unsupported_input(
    mocker,
):
    num_macs = NeutronTargetSpec("imxrt700", "SDK_25_09").get_num_macs()
    model = Conv2dModule(
        in_channels=num_macs,
        out_channels=num_macs,  # The output `Transpose` will be supported.
        padding=1,
        stride=1,
    )
    input_shape = (1, num_macs, num_macs, 3)  # The input `Transpose` is not supported.

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    payload_header_spy = mocker.spy(PayloadComposer, "_create_payload_header")
    edge_program = to_quantized_edge_program(
        model,
        input_shape,
        use_neutron_for_format_conversion=True,  # Make sure the IR converter inserts the extra `Transpose` operators.
    ).exported_program()

    # Make sure the edge_program only contains the 1 delegate call.
    nodes = list(edge_program.graph.nodes)
    assert len(nodes) == 7
    assert "call_delegate" in nodes[3].name
    assert not graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.aten.convolution.default]
    )
    assert not graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.aten.permute_copy.default]
    )

    # Capture the converted IR model.
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Make sure there is just the 1 `Transpose` in the model.
    tflite_subgraph = Model.GetRootAs(tflite_flatbuffers_model).Subgraphs(0)
    assert tflite_subgraph.OperatorsLength() == 3
    assert (
        tflite_subgraph.Operators(0).BuiltinOptionsType() == BuiltinOptions.PadV2Options
    )
    assert (
        tflite_subgraph.Operators(1).BuiltinOptionsType()
        == BuiltinOptions.Conv2DOptions
    )
    assert (
        tflite_subgraph.Operators(2).BuiltinOptionsType()
        == BuiltinOptions.TransposeOptions
    )

    # Get the header of the payload for the delegated partition.
    payload_header = payload_header_spy.spy_return
    assert payload_header.size == 7
    # the 4th and 5th bytes indicate the format. `1` means `channels_last`, which means the runtime will transpose the data.
    assert all(payload_header[3:5] == [1, 0])  # [<input_byte>, <output_byte>]


def test_delegating_format_related_transpose_operators__supported_input__unsupported_output(
    mocker,
):
    num_macs = NeutronTargetSpec("imxrt700", "SDK_25_09").get_num_macs()
    model = Conv2dModule(
        in_channels=num_macs,
        out_channels=3,  # The output `Transpose` will NOT be supported.
        stride=1,
    )
    input_shape = (1, num_macs, 3, num_macs)  # The input `Transpose` is supported.

    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    payload_header_spy = mocker.spy(PayloadComposer, "_create_payload_header")
    edge_program = to_quantized_edge_program(
        model,
        input_shape,
        use_neutron_for_format_conversion=True,  # Make sure the IR converter inserts the extra `Transpose` operators.
    ).exported_program()

    # Make sure the edge_program only contains the 1 delegate call.
    nodes = list(edge_program.graph.nodes)
    assert len(nodes) == 7
    assert "call_delegate" in nodes[3].name
    assert not graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.aten.convolution.default]
    )
    assert not graph_contains_any_of_ops(
        edge_program.graph, [torch.ops.aten.permute_copy.default]
    )

    # Capture the converted IR model.
    tflite_flatbuffers_model, _ = converter_spy.spy_return

    # Make sure there is just the 1 `Transpose` in the model.
    tflite_subgraph = Model.GetRootAs(tflite_flatbuffers_model).Subgraphs(0)
    assert tflite_subgraph.OperatorsLength() == 2
    assert (
        tflite_subgraph.Operators(0).BuiltinOptionsType()
        == BuiltinOptions.TransposeOptions
    )
    assert (
        tflite_subgraph.Operators(1).BuiltinOptionsType()
        == BuiltinOptions.Conv2DOptions
    )

    # Get the header of the payload for the delegated partition.
    payload_header = payload_header_spy.spy_return
    assert payload_header.size == 7
    # the 4th and 5th bytes indicate the format. `1` means `channels_last`, which means the runtime will transpose the data.
    assert all(payload_header[3:5] == [0, 1])  # [<input_byte>, <output_byte>]
