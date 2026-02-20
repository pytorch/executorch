# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    is_annotated,
    QuantizationConfig,
    TOSAQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import QuantizationSpec
from executorch.backends.arm.tosa import TosaSpecification
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

DQ_PER_CHANNEL = torch.ops.quantized_decomposed.dequantize_per_channel.default
DQ_PER_TENSOR = torch.ops.quantized_decomposed.dequantize_per_tensor.default
Q_PER_TENSOR = torch.ops.quantized_decomposed.quantize_per_tensor.default


class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(
            3,
            16,
            kernel_size=4,
        )
        self.conv1 = torch.nn.Conv2d(16, 32, kernel_size=3, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv0(x)
        x = torch.sigmoid(x)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        return x


test_inputs = (torch.randn(1, 3, 64, 64),)


def validate_per_tensor_quant(node: torch.fx.Node, qspec: QuantizationSpec):
    _, _, zero_point, qmin, qmax, dtype = node.args
    if qspec.qscheme == torch.per_tensor_symmetric:
        assert (
            zero_point == 0
        ), f"Zero point {zero_point} is not zero for symmetric quantization"
    assert (
        qmin == qspec.quant_min
    ), f"Quant min {qmin} does not match expected {qspec.quant_min}"
    assert (
        qmax == qspec.quant_max
    ), f"Quant max {qmax} does not match expected {qspec.quant_max}"
    assert dtype == qspec.dtype, f"Dtype {dtype} does not match expected {qspec.dtype}"


def validate_per_channel_quant(node: torch.fx.Node, qspec: QuantizationSpec):
    _, _, _, channel_axis, qmin, qmax, dtype = node.args
    assert (
        channel_axis == qspec.ch_axis
    ), f"Channel axis {channel_axis} does not match expected {qspec.ch_axis}"
    assert (
        qmin == qspec.quant_min
    ), f"Quant min {qmin} does not match expected {qspec.quant_min}"
    assert (
        qmax == qspec.quant_max
    ), f"Quant max {qmax} does not match expected {qspec.quant_max}"
    assert dtype == qspec.dtype, f"Dtype {dtype} does not match expected {qspec.dtype}"


def validate_input(input_node: torch.fx.Node, qspec: QuantizationSpec | None):
    if qspec is None:
        return

    per_channel = qspec.qscheme == torch.per_channel_symmetric
    expected_dequant_op = DQ_PER_CHANNEL if per_channel else DQ_PER_TENSOR
    assert (
        input_node.target == expected_dequant_op
    ), f"Input node {input_node} is not quantized as expected"
    if per_channel:
        validate_per_channel_quant(input_node, qspec)
    else:
        validate_per_tensor_quant(input_node, qspec)


def validate_output(node: torch.fx.Node, qspec: QuantizationSpec | None):
    if qspec is None:
        return
    users = list(node.users)
    assert len(users) == 1, f"Node {node} should have exactly one user"
    assert (
        users[0].target == Q_PER_TENSOR
    ), f"Output node {users[0]} is not quantized as expected"
    validate_per_tensor_quant(users[0], qspec)


def validate_node(
    node: torch.fx.Node, quantization_config: QuantizationConfig | None
) -> None:
    if quantization_config is None:
        assert not is_annotated(node), f"Node {node} is unexpectedly annotated"
        return

    assert is_annotated(node), f"Node {node} is not annotated"
    input_qspec = quantization_config.get_input_act_qspec()
    output_qspec = quantization_config.get_output_act_qspec()
    weight_qspec = quantization_config.get_weight_qspec()

    if len(node.all_input_nodes) == 3:
        input_node, weight_node, bias_node = node.all_input_nodes
        bias_qspec = quantization_config.get_bias_qspec(node)
        validate_input(bias_node, bias_qspec)
    else:
        input_node, weight_node = node.all_input_nodes

    validate_input(input_node, input_qspec)
    validate_input(weight_node, weight_qspec)
    validate_output(node, output_qspec)


def test_set_module_name_tosa_INT() -> None:
    model = ConvModel()
    model.eval()

    # Set up quantizer with different configs for different modules
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(tosa_spec)
    int8_config = get_symmetric_quantization_config(is_per_channel=False)
    a16w8_config = get_symmetric_a16w8_quantization_config()
    # Set module-specific configurations but don't set global config to test that
    # only specified modules are quantized
    quantizer.set_module_name("conv0", int8_config)
    quantizer.set_module_name("conv1", a16w8_config)

    # Export model
    exported_model = torch.export.export(model, test_inputs)

    # Prepare, calibrate and convert model
    prepared_model = prepare_pt2e(exported_model.module(), quantizer)
    prepared_model(*test_inputs)
    converted_model = convert_pt2e(prepared_model)

    validate_node(
        [node for node in converted_model.graph.nodes if node.name == "conv2d"][0],
        int8_config,
    )
    validate_node(
        [node for node in converted_model.graph.nodes if node.name == "conv2d_1"][0],
        a16w8_config,
    )
    validate_node(
        [node for node in converted_model.graph.nodes if node.name == "conv2d_2"][0],
        None,
    )
