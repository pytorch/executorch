# Copyright 2024 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests for NeutronQuantizer.

import executorch.backends.nxp.tests.models as models
import torch
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def _get_target_name(node):
    return node._pretty_print_target(node.target)


def test_quantizer_conv2d():
    model = models.Conv2dModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export_for_training(
        model, example_input, strict=True
    ).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 11
    assert nodes[7].name == "conv2d"
    # [0]: Input, [1] : weights, [2]: bias
    assert (
        _get_target_name(nodes[7].args[0])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[7].args[1])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[7].args[2])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[8])
        == "torch.ops.quantized_decomposed.quantize_per_tensor.default"
    )
    assert nodes[8].args[0].name == "conv2d"


def test_quantizer_linear():
    model = models.LinearModule(bias=True)
    model.eval()

    example_input = (torch.ones(10, 32),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export_for_training(
        model, example_input, strict=True
    ).module()

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
        _get_target_name(nodes[7].args[0])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[7].args[1])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[7].args[2])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[8])
        == "torch.ops.quantized_decomposed.quantize_per_tensor.default"
    )
    assert nodes[8].args[0].name == "linear"


def test_quantizer_maxpool2d():
    model = models.Conv2dAndMaxPool2DModule()
    model.eval()

    example_input = (torch.ones(1, 8, 32, 32),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export_for_training(
        model, example_input, strict=True
    ).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 14
    # Check if QDQ pattern:
    assert nodes[10].name == "max_pool2d"
    assert (
        _get_target_name(nodes[10].args[0])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[11])
        == "torch.ops.quantized_decomposed.quantize_per_tensor.default"
    )
    assert nodes[11].args[0].name == "max_pool2d"

    # Check if input and output quantization is same
    input_quant = nodes[10].args[0].args[1:]
    output_quant = nodes[11].args[1:]
    assert input_quant == output_quant


def test_quantizer_softmax():
    model = models.SoftmaxModule(dim=0)
    model.eval()

    example_input = (torch.ones(1, 10),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export_for_training(
        model, example_input, strict=True
    ).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 7
    # Check if QDQ pattern:
    assert nodes[3].name == "softmax"
    assert (
        _get_target_name(nodes[3].args[0])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[4])
        == "torch.ops.quantized_decomposed.quantize_per_tensor.default"
    )
    assert nodes[4].args[0].name == "softmax"

    # Check output quantization
    scale, zp, _, _, dtype = nodes[4].args[1:]
    assert scale == 1.0 / 256.0
    assert zp == -128
    assert dtype == torch.int8


def test_quantizer_single_maxpool2d():
    model = models.MaxPool2dModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export_for_training(
        model, example_input, strict=True
    ).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 3
    assert nodes[1].name == "max_pool2d"
    assert "quantization_annotation" not in nodes[1].meta


def test_quantizer_conv2d_relu():
    model = models.Conv2dReLUModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export_for_training(
        model, example_input, strict=True
    ).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 12
    assert nodes[7].name == "dequantize_per_tensor_default_2"
    assert nodes[8].name == "relu"
    assert nodes[9].name == "quantize_per_tensor_default_3"


def test_quantizer_conv2d_avg_pool2d():
    model = models.AvgPool2dConvModule(count_include_pad=False)
    model.eval()

    example_input = (torch.ones(1, 4, 16, 16),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export_for_training(
        model, example_input, strict=True
    ).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 14
    assert nodes[9].name == "dequantize_per_tensor_default_3"
    assert nodes[10].name == "avg_pool2d"
    assert nodes[11].name == "quantize_per_tensor_default_4"


def test_quantizer_conv2d_permute():
    model = models.Conv2dPermuteModule()
    model.eval()

    example_input = (torch.ones(1, 4, 16, 16),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export_for_training(
        model, example_input, strict=True
    ).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 12
    assert nodes[7].name == "dequantize_per_tensor_default_2"
    assert nodes[8].name == "permute"
    assert nodes[9].name == "quantize_per_tensor_default_3"
