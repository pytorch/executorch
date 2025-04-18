# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.export import ExportedProgram

from executorch.backends.nxp.backend.edge_program_converter import EdgeProgramToIRConverter
from executorch.backends.nxp.backend.ir.converter.node_converter import Target
from executorch.backends.nxp.backend.ir.lib.tflite.BuiltinOptions import BuiltinOptions
from executorch.backends.nxp.backend.ir.lib.tflite.Model import Model
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import TFLiteExecutor, EdgeProgramExecutor, convert_run_compare, \
    ToNHWCPreprocess
from executorch.backends.nxp.tests.models import Conv2dModule, LinearSoftmaxModule
from executorch.backends.nxp.tests.models import ConvFCSoftmaxModule


def test_neutron_backend__single_conv_model():
    edge_program_manager = to_quantized_edge_program(Conv2dModule(bias=False), (1, 4, 32, 32))
    lowered_module = edge_program_manager.exported_program().graph_module.lowered_module_0
    assert len(lowered_module.processed_bytes) != 0  # The Neutron microcode, weights and kernels have been written here


def test_neutron_backend__single_conv_model__payload_header_channels_last():
    edge_program_manager = to_quantized_edge_program(Conv2dModule(bias=False), (1, 4, 32, 32))
    payload = edge_program_manager.exported_program().graph_module.lowered_module_0.processed_bytes

    assert payload[0] == 0x1  # Single input
    assert payload[1] == 0x1  # Single output
    assert payload[2] == 0x1  # Channels last
    assert payload[3] == 0x1  # Channels last
    assert all(byte == 0x0 for byte in payload[4:16])  # Aligned to 16 bytes
    assert payload[17] != 0x0  # Followed by non-zero content


def test_neutron_backend__linear_softmax_model__payload_header_formatless():
    edge_program_manager = to_quantized_edge_program(LinearSoftmaxModule(), (1, 12))
    payload = edge_program_manager.exported_program().graph_module.lowered_module_0.processed_bytes

    assert payload[0] == 0x1  # Single input
    assert payload[1] == 0x1  # Single output
    assert payload[2] == 0x0  # Formatless
    assert payload[3] == 0x0  # Formatless
    assert all(byte == 0x0 for byte in payload[4:16])  # Aligned to 16 bytes
    assert payload[17] != 0x0  # Followed by non-zero content


def test_lowered_program_and_tflite_output_match__conv2d__no_bias(mocker):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")

    model = Conv2dModule(bias=False)
    input_shape = (1, 4, 32, 32)

    # Run conversion
    prog = to_quantized_edge_program(model, input_shape)

    # Capture generated model
    tflite_flatbuffers_model, io_formats = converter_spy.spy_return

    tflite_model = Model.GetRootAs(tflite_flatbuffers_model)
    sub_graph = tflite_model.Subgraphs(0)

    assert sub_graph.OperatorsLength() == 1
    assert sub_graph.Operators(0).BuiltinOptionsType() == BuiltinOptions.Conv2DOptions

    # Capture converted program
    exported_program: ExportedProgram = converter_spy.call_args.args[1]

    input_data = (torch.randn(input_shape, dtype=torch.float32) * 50).type(torch.int8).detach().numpy()
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
    assert tflite_subgraph.Operators(0).BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
    assert tflite_subgraph.Operators(1).BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
    assert tflite_subgraph.Operators(2).BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions

    # Verify outputs of program and TFLite model
    input_data = (torch.randn(input_shape, dtype=torch.float32)).type(torch.int8).detach().numpy()
    convert_run_compare(exported_program, input_data=input_data, tflite_input_preprocess=ToNHWCPreprocess())
