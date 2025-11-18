# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests for NeutronQuantizer.

from copy import deepcopy

import executorch.backends.nxp.tests.executorch_pipeline as executorch_pipeline
import executorch.backends.nxp.tests.models as models
import numpy as np
import pytest
import torch

from executorch.backends.nxp.backend.edge_program_converter import (
    EdgeProgramToIRConverter,
)

from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.tests.executorch_pipeline import (
    neutron_target_spec,
    to_quantized_edge_program,
)
from executorch.backends.nxp.tests.executors import (
    convert_run_compare,
    graph_contains_any_of_ops,
    ToChannelFirstPreprocess,
    ToChannelLastPreprocess,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

fuse_activation_ops = [
    exir_ops.edge.aten.addmm.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.tanh.default,
]


def test_quantizer_conv2d():
    model = models.Conv2dModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 15
    assert nodes[11].name == "conv2d"
    # [0]: Input, [1] : weights, [2]: bias
    assert (
        nodes[11].args[0].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert (
        nodes[11].args[1].target
        == torch.ops.quantized_decomposed.dequantize_per_channel.default
    )
    assert (
        nodes[11].args[2].target
        == torch.ops.quantized_decomposed.dequantize_per_channel.default
    )
    assert (
        nodes[12].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )
    assert nodes[12].args[0].target == torch.ops.aten.conv2d.default


def test_quantizer_linear():
    model = models.LinearModule(bias=True)
    model.eval()

    example_input = (torch.ones(10, 32),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 11
    assert nodes[7].name == "linear"
    # [0]: Input, [1] : weights, [2]: bias
    assert (
        nodes[7].args[0].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert (
        nodes[7].args[1].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert (
        nodes[7].args[2].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert nodes[8].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    assert nodes[8].args[0].target == torch.ops.aten.linear.default


def test_quantizer_maxpool2d():
    model = models.Conv2dAndMaxPool2DModule()
    model.eval()

    example_input = (torch.ones(1, 8, 32, 32),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 18
    # Check if QDQ pattern:
    assert nodes[14].target == torch.ops.aten.max_pool2d.default
    assert (
        nodes[14].args[0].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert (
        nodes[15].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )
    assert nodes[15].args[0].target == torch.ops.aten.max_pool2d.default

    # Check if input and output quantization is same
    input_quant = nodes[14].args[0].args[1:]
    output_quant = nodes[15].args[1:]
    assert input_quant == output_quant


def test_quantizer_softmax():
    model = models.SoftmaxModule(dim=0)
    model.eval()

    example_input = (torch.ones(1, 10),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 7
    # Check if QDQ pattern:
    assert nodes[3].target == torch.ops.aten.softmax.int
    assert (
        nodes[3].args[0].target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert nodes[4].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    assert nodes[4].args[0].target == torch.ops.aten.softmax.int

    # Check output quantization
    scale, zp, _, _, dtype = nodes[4].args[1:]
    assert scale == 1.0 / 256.0
    assert zp == -128
    assert dtype == torch.int8


def test_quantizer_single_maxpool2d():
    model = models.MaxPool2dModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 7
    assert nodes[3].target == torch.ops.aten.max_pool2d.default
    assert "quantization_annotation" not in nodes[1].meta


def test_quantizer_conv2d_relu():
    model = models.Conv2dReLUModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)

    assert len(nodes) == 12
    assert (
        nodes[6].target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert nodes[7].target == torch.ops.aten.conv2d.default
    assert nodes[8].target == torch.ops.aten.relu.default
    assert nodes[9].target == torch.ops.quantized_decomposed.quantize_per_tensor.default


def test_quantizer_conv2d_avg_pool2d():
    model = models.AvgPool2dConvModule(count_include_pad=False)
    model.eval()

    example_input = (torch.ones(1, 4, 16, 16),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)

    assert len(nodes) == 18
    assert (
        nodes[13].target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert nodes[14].target == torch.ops.aten.avg_pool2d.default
    assert (
        nodes[15].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )


def test_quantizer_conv2d_permute():
    model = models.Conv2dPermuteModule()
    model.eval()

    example_input = (torch.ones(1, 4, 16, 16),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)

    assert len(nodes) == 14
    assert (
        nodes[9].target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert nodes[10].target == torch.ops.aten.permute.default
    assert (
        nodes[11].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )


def test_multiple_shared_spec_ops_in_row():
    """
    This test demonstrates that having two operators in a row, both relying on quantizers
    with SharedSpecPattern, does not break the quantization process.
    """
    model = models.Conv2dReLUMaxPoolModule()
    model.eval()

    example_input = (torch.ones(1, 3, 64, 64),)
    quantizer = NeutronQuantizer(neutron_target_spec)
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)

    assert len(nodes) == 15
    assert (
        nodes[-5].target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert nodes[-4].target == torch.ops.aten.max_pool2d.default
    assert (
        nodes[-3].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )

    # Assert that post-ReLU quantize and pre-MaxPool dequantize has same specs
    assert nodes[-6].args[1:] == nodes[-5].args[1:]
    # Assert that post-Conv quantize and pre-ReLU dequantize has same specs
    assert nodes[5].args[1:] == nodes[6].args[1:]


def test_quantizers_order_invariance():
    """
    This test demonstrates that the order of quantizers in NeutronQuantizer
    does not affect the resulting graph.
    """
    model = models.Conv2dReLUModule()
    model.eval()

    example_input = (torch.ones(1, 4, 64, 64),)
    quantizer = NeutronQuantizer(neutron_target_spec)

    graph_module = torch.export.export(model, example_input, strict=True).module()

    m = prepare_pt2e(deepcopy(graph_module), quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    quantizer.quantizers = quantizer.quantizers[::-1]
    m_reversed = prepare_pt2e(graph_module, quantizer)
    m_reversed(*example_input)
    m_reversed = convert_pt2e(m)

    # Dry run
    m(*example_input)
    m_reversed(*example_input)

    nodes = list(m.graph.nodes)
    nodes_reversed = list(m.graph.nodes)

    assert len(nodes) == len(nodes_reversed)
    assert all(n == n_reversed for n, n_reversed in zip(nodes, nodes_reversed))


@pytest.mark.parametrize(
    "activation, inplace",
    [
        ("relu", True),
        ("relu", False),
        ("relu6", True),
        ("relu6", False),
        ("tanh", True),
        ("tanh", False),
        ("sigmoid", False),
    ],
)
def test_quantizer__linear_w_activation(mocker, activation, inplace):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    quantizer_spy = mocker.spy(executorch_pipeline, "calibrate_and_quantize")

    input_shape = (1, 4)
    model = models.LinearActivationModule(
        activation=activation,
        inplace=inplace,
        in_channels=input_shape[1],
        mode="linear",
    )

    edge_program = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure that all nodes were delegated.
    assert not graph_contains_any_of_ops(
        graph=edge_program.graph,
        ops=fuse_activation_ops,
    )
    assert any("lowered_module" in node.name for node in edge_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    exir_program_aten_quant: GraphModule = quantizer_spy.spy_return

    # Check linear and activation are in the same QDQ cluster
    nodes = list(exir_program_aten_quant.graph.nodes)
    assert len(nodes) == 12
    assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
        nodes[7]
    )
    assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
        nodes[8]
    )
    assert nodes[9].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        input_data,
        tfl_model=tflite_flatbuffers_model,
        atol=1.0,
    )


@pytest.mark.parametrize(
    "activation, inplace",
    [
        ("relu", True),
        ("relu", False),
        ("relu6", True),
        ("relu6", False),
        ("tanh", True),
        ("tanh", False),
        ("sigmoid", False),
    ],
)
def test_quantizer__addmm_w_activation(mocker, activation, inplace):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    quantizer_spy = mocker.spy(executorch_pipeline, "calibrate_and_quantize")

    input_shape = (1, 4)
    model = models.LinearActivationModule(
        activation=activation, inplace=inplace, in_channels=input_shape[1], mode="addmm"
    )

    edge_program = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure that all nodes were delegated.
    assert not graph_contains_any_of_ops(
        graph=edge_program.graph,
        ops=fuse_activation_ops,
    )
    assert any("lowered_module" in node.name for node in edge_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    exir_program_aten_quant: GraphModule = quantizer_spy.spy_return

    # Check linear and activation are in the same QDQ cluster
    nodes = list(exir_program_aten_quant.graph.nodes)
    assert len(nodes) == 12
    assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
        nodes[7]
    )
    assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
        nodes[8]
    )
    assert nodes[9].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        input_data,
        tfl_model=tflite_flatbuffers_model,
        atol=1.0,
    )


@pytest.mark.parametrize(
    "activation, inplace",
    [
        ("relu", True),
        ("relu", False),
        ("relu6", True),
        ("relu6", False),
        ("tanh", True),
        ("tanh", False),
        ("sigmoid", False),
    ],
)
def test_quantizer__mm_w_activation(mocker, activation, inplace):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    quantizer_spy = mocker.spy(executorch_pipeline, "calibrate_and_quantize")

    input_shape = (1, 4)
    model = models.LinearActivationModule(
        activation=activation, inplace=inplace, in_channels=input_shape[1], mode="mm"
    )

    edge_program = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure that all nodes were delegated.
    assert not graph_contains_any_of_ops(
        graph=edge_program.graph,
        ops=fuse_activation_ops,
    )
    assert any("lowered_module" in node.name for node in edge_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    exir_program_aten_quant: GraphModule = quantizer_spy.spy_return

    # Check linear and activation are in the same QDQ cluster
    nodes = list(exir_program_aten_quant.graph.nodes)
    assert len(nodes) == 10
    assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
        nodes[5]
    )
    assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
        nodes[6]
    )
    assert nodes[7].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        input_data,
        tfl_model=tflite_flatbuffers_model,
        atol=1.0,
    )


@pytest.mark.parametrize(
    "activation, inplace",
    [
        ("relu", True),
        ("relu", False),
        ("relu6", True),
        ("relu6", False),
        ("tanh", True),
        ("tanh", False),
        ("sigmoid", False),
    ],
)
def test_quantizer__conv_w_activation(mocker, activation, inplace):
    converter_spy = mocker.spy(EdgeProgramToIRConverter, "convert_program")
    quantizer_spy = mocker.spy(executorch_pipeline, "calibrate_and_quantize")

    input_shape = (1, 4, 8, 8)
    model = models.ConvActivationModule(
        activation=activation, inplace=inplace, in_channels=input_shape[1]
    )

    edge_program = to_quantized_edge_program(model, input_shape).exported_program()

    # Make sure that all nodes were delegated.
    assert not graph_contains_any_of_ops(
        graph=edge_program.graph,
        ops=fuse_activation_ops,
    )
    assert any("lowered_module" in node.name for node in edge_program.graph.nodes)

    tflite_flatbuffers_model, io_formats = converter_spy.spy_return
    exported_program: ExportedProgram = converter_spy.call_args.args[1]
    exir_program_aten_quant: GraphModule = quantizer_spy.spy_return

    # Check linear and activation are in the same QDQ cluster
    nodes = list(exir_program_aten_quant.graph.nodes)
    assert len(nodes) == 16
    assert neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
        nodes[11]
    )
    assert neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
        nodes[12]
    )
    assert (
        nodes[13].target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )
    input_data = (np.random.random(input_shape).astype(np.float32) * 50).astype(np.int8)
    convert_run_compare(
        exported_program,
        input_data,
        tfl_model=tflite_flatbuffers_model,
        tflite_input_preprocess=ToChannelLastPreprocess(),
        tflite_output_preprocess=ToChannelFirstPreprocess(),
        atol=1.0,
    )
