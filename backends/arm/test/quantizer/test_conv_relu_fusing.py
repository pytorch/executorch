# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import (
    QuantizationConfig,
    QuantizationSpec,
)
from executorch.backends.arm.test.tester.test_pipeline import QuantizationPipeline
from executorch.backends.arm.tosa import TosaSpecification


def get_symmetric_a8w8_quantization_config():
    affine_quant_config = get_symmetric_quantization_config()
    output_activation = QuantizationSpec(
        dtype=torch.int8,
        observer_or_fake_quant_ctr=affine_quant_config.get_output_act_qspec().observer_or_fake_quant_ctr,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=None,
        is_dynamic=False,
    )
    input_activation = output_activation
    symmetric_quant_config = QuantizationConfig(
        input_activation=input_activation,
        output_activation=output_activation,
        weight=affine_quant_config.get_weight_qspec(),
        bias=None,
    )
    return symmetric_quant_config


class ConvBNRelu(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=2,
        )
        self.bn = torch.nn.BatchNorm2d(num_features=4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu

    def get_example_inputs(self):
        return (torch.randn(1, 3, 8, 8),)


def test_conv_relu_fusing_8a8w_tosa_INT_affine():
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(tosa_spec)
    quant_config = get_symmetric_quantization_config()
    quantizer.set_global(quant_config)
    expected_annotations = {
        "aten.conv2d.default": {None: 1},
        "aten.relu.default": {quant_config.get_output_act_qspec(): 1},
    }
    pipeline = QuantizationPipeline[Tuple[torch.Tensor]](
        ConvBNRelu(),
        ConvBNRelu().get_example_inputs(),
        quantizer=quantizer,
        qspecs=expected_annotations,
    )
    pipeline.run()


def test_conv_relu_fusing_8a8w_tosa_INT_symmetric():
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(tosa_spec)
    symmetric_quant_config = get_symmetric_a8w8_quantization_config()

    quantizer.set_global(symmetric_quant_config)
    expected_annotations = {
        "aten.conv2d.default": {symmetric_quant_config.get_output_act_qspec(): 1},
        "aten.relu.default": {symmetric_quant_config.get_output_act_qspec(): 1},
    }
    pipeline = QuantizationPipeline[Tuple[torch.Tensor]](
        ConvBNRelu(),
        ConvBNRelu().get_example_inputs(),
        quantizer=quantizer,
        qspecs=expected_annotations,
    )
    pipeline.run()


def test_conv_relu_fusing_16a8w_tosa_INT_symmetric():
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT+int16")
    quantizer = TOSAQuantizer(tosa_spec)
    quant_config = get_symmetric_a16w8_quantization_config()

    quantizer.set_global(quant_config)
    expected_annotations = {
        "aten.conv2d.default": {quant_config.get_output_act_qspec(): 1},
        "aten.relu.default": {quant_config.get_output_act_qspec(): 1},
    }
    pipeline = QuantizationPipeline[Tuple[torch.Tensor]](
        ConvBNRelu(),
        ConvBNRelu().get_example_inputs(),
        quantizer=quantizer,
        qspecs=expected_annotations,
    )
    pipeline.run()
