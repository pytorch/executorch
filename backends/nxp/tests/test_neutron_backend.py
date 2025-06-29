# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.Model import Model
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.nxp_backend import PayloadComposer
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.backends.nxp.tests.models import Conv2dModule, LinearSoftmaxModule


def test_neutron_backend__single_conv_model():
    edge_program_manager = to_quantized_edge_program(
        Conv2dModule(bias=False), (1, 4, 32, 32)
    )
    lowered_module = (
        edge_program_manager.exported_program().graph_module.lowered_module_0
    )
    assert (
        len(lowered_module.processed_bytes) != 0
    )  # The Neutron microcode, weights and kernels have been written here


def test_neutron_backend__single_conv_model__payload_header_channels_last():
    edge_program_manager = to_quantized_edge_program(
        Conv2dModule(bias=False),
        (1, 4, 32, 32),
        use_neutron_for_format_conversion=False,
    )
    payload = (
        edge_program_manager.exported_program().graph_module.lowered_module_0.processed_bytes
    )

    assert payload[0] == 0x1  # Number of Neutron node inputs
    assert payload[1] == 0x1  # Number of Neutron node outputs
    assert payload[2] == 0x1  # Number of model inputs
    assert payload[3] == 0x1  # Channels last 0-th Neutron input
    assert payload[4] == 0x1  # Channels last 0-th Neutron output
    assert payload[5] == 0x0  # Map 0-th Neutron input to 0-th model input
    assert payload[6] == 0x0  # Map 0-th Neutron output to 0-th model output
    assert all(byte == 0x0 for byte in payload[7:16])  # Aligned to 16 bytes
    assert payload[17] != 0x0  # Followed by non-zero content


def test_neutron_backend__linear_softmax_model__payload_header_formatless():
    edge_program_manager = to_quantized_edge_program(LinearSoftmaxModule(), (1, 12))
    payload = (
        edge_program_manager.exported_program().graph_module.lowered_module_0.processed_bytes
    )

    assert payload[0] == 0x1  # Number of Neutron node inputs
    assert payload[1] == 0x1  # Number of Neutron node outputs
    assert payload[2] == 0x1  # Number of model inputs
    assert payload[3] == 0x0  # Formatless 0-th Neutron input
    assert payload[4] == 0x0  # Formatless 0-th Neutron output
    assert payload[5] == 0x0  # Map 0-th Neutron input to 0-th model input
    assert payload[6] == 0x0  # Map 0-th Neutron output to 0-th model output
    assert all(byte == 0x0 for byte in payload[7:16])  # Aligned to 16 bytes
    assert payload[17] != 0x0  # Followed by non-zero content


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
