# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests for NeutronQuantizer.

from copy import deepcopy

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
        _get_target_name(nodes[11].args[0])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[11].args[1])
        == "torch.ops.quantized_decomposed.dequantize_per_channel.default"
    )
    assert (
        _get_target_name(nodes[11].args[2])
        == "torch.ops.quantized_decomposed.dequantize_per_channel.default"
    )
    assert (
        _get_target_name(nodes[12])
        == "torch.ops.quantized_decomposed.quantize_per_tensor.default"
    )
    assert nodes[12].args[0].name == "conv2d"


def test_quantizer_linear():
    model = models.LinearModule(bias=True)
    model.eval()

    example_input = (torch.ones(10, 32),)
    quantizer = NeutronQuantizer()
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
    assert nodes[14].name == "max_pool2d"
    assert (
        _get_target_name(nodes[14].args[0])
        == "torch.ops.quantized_decomposed.dequantize_per_tensor.default"
    )
    assert (
        _get_target_name(nodes[15])
        == "torch.ops.quantized_decomposed.quantize_per_tensor.default"
    )
    assert nodes[15].args[0].name == "max_pool2d"

    # Check if input and output quantization is same
    input_quant = nodes[14].args[0].args[1:]
    output_quant = nodes[15].args[1:]
    assert input_quant == output_quant


def test_quantizer_single_maxpool2d():
    model = models.MaxPool2dModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 7
    assert nodes[3].name == "max_pool2d"
    assert "quantization_annotation" not in nodes[1].meta


def test_quantizer_conv2d_relu():
    model = models.Conv2dReLUModule()
    model.eval()

    example_input = (torch.ones(1, 4, 32, 32),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 14
    assert nodes[9].name == "dequantize_per_tensor_default_1"
    assert nodes[10].name == "relu"
    assert nodes[11].name == "quantize_per_tensor_default_2"


def test_quantizer_conv2d_avg_pool2d():
    model = models.AvgPool2dConvModule(count_include_pad=False)
    model.eval()

    example_input = (torch.ones(1, 4, 16, 16),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)
    assert len(nodes) == 18
    assert nodes[13].name == "dequantize_per_tensor_default_1"
    assert nodes[14].name == "avg_pool2d"
    assert nodes[15].name == "quantize_per_tensor_default_2"


def test_quantizer_conv2d_permute():
    model = models.Conv2dPermuteModule()
    model.eval()

    example_input = (torch.ones(1, 4, 16, 16),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)

    assert len(nodes) == 14
    assert nodes[9].name == "dequantize_per_tensor_default_1"
    assert nodes[10].name == "permute"
    assert nodes[11].name == "quantize_per_tensor_default_2"


def test_multiple_shared_spec_ops_in_row():
    """
    This test demonstrates that having two operators in a row, both relying on quantizers
    with SharedSpecPattern, does not break the quantization process.
    """
    model = models.Conv2dReLUMaxPoolModule()
    model.eval()

    example_input = (torch.ones(1, 3, 64, 64),)
    quantizer = NeutronQuantizer()
    graph_module = torch.export.export(model, example_input, strict=True).module()

    # noinspection PyTypeChecker
    m = prepare_pt2e(graph_module, quantizer)
    m(*example_input)
    m = convert_pt2e(m)

    # Dry run
    m(*example_input)

    nodes = list(m.graph.nodes)

    assert len(nodes) == 17
    assert nodes[-5].name.startswith("dequantize_per_tensor_default")
    assert nodes[-4].name == "max_pool2d"
    assert nodes[-3].name.startswith("quantize_per_tensor_default")

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
    quantizer = NeutronQuantizer()

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
