# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree..

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from executorch.backends.nxp.quantizer.utils import calibrate_and_quantize
from executorch.backends.transforms.quantize_fused_convbn_bias_pass import (
    QuantizeFusedConvBnBiasPass,
)
from executorch.backends.xnnpack.test.tester.tester import Quantize
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch import nn
from torch.export import export


input_t = Tuple[torch.Tensor]


class ConvBnNoBias(nn.Module):
    """Conv2d with bias=False followed by BatchNorm. QAT fusion introduces a bias."""

    def __init__(self, per_channel: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(16)

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 32, 32),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class ConvBnReluNoBias(nn.Module):
    """Conv2d with bias=False, BatchNorm, and ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 32, 32),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1dBnNoBias(nn.Module):
    """Conv1d with bias=False followed by BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(3, 8, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm1d(8)

    def get_inputs(self) -> input_t:
        return (torch.randn(2, 3, 16),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


# --- ARM (TOSA) tests ---

arm_models = {
    "conv2d_bn_no_bias_per_channel": (ConvBnNoBias(), True),
    "conv2d_bn_no_bias_per_tensor": (ConvBnNoBias(), False),
    "conv2d_bn_relu_no_bias_per_channel": (ConvBnReluNoBias(), True),
    "conv2d_bn_relu_no_bias_per_tensor": (ConvBnReluNoBias(), False),
    "conv1d_bn_no_bias_per_channel": (Conv1dBnNoBias(), True),
    "conv1d_bn_no_bias_per_tensor": (Conv1dBnNoBias(), False),
}


@common.parametrize("test_data", arm_models)
def test_quantize_fused_convbn_bias_arm_qat(test_data) -> None:
    """
    Test that QuantizeFusedConvBnBiasPass correctly quantizes the bias
    introduced by BatchNorm fusion during QAT when the original conv has bias=False.
    Uses the ARM TOSA quantizer.
    """
    model, per_channel = test_data
    pipeline = PassPipeline[input_t](
        model,
        model.get_inputs(),
        quantize=True,
        passes_with_exported_program=[QuantizeFusedConvBnBiasPass],
    )

    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    pipeline.change_args(
        "quantize",
        Quantize(
            quantizer=quantizer,
            quantization_config=get_symmetric_quantization_config(
                is_qat=True, is_per_channel=per_channel
            ),
        ),
    )

    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


# --- NXP (Neutron) tests ---


def _run_nxp_qat_pass(model: nn.Module, use_edge: bool = True) -> None:
    """Quantize a model with NXP's NeutronQuantizer in QAT mode, optionally convert to
    edge, and verify that QuantizeFusedConvBnBiasPass quantizes the fused bias."""
    example_input = model.get_inputs()

    target_spec = NeutronTargetSpec(
        target="imxrt700", neutron_converter_flavor="SDK_25_12"
    )
    quantizer = NeutronQuantizer(target_spec, is_qat=True)

    exported = export(model, example_input, strict=True)
    quantized_model = calibrate_and_quantize(
        model=exported,
        calibration_inputs=[example_input],
        quantizer=quantizer,
        is_qat=True,
    )

    exported_program = export(quantized_model, example_input, strict=True)

    if use_edge:
        edge_program_manager = to_edge(exported_program)
        exported_program = edge_program_manager.exported_program()
        conv_targets = (exir_ops.edge.aten.convolution.default,)
        dequant_targets = (
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        )
    else:
        conv_targets = (
            torch.ops.aten.convolution.default,
            torch.ops.aten.conv2d.default,
        )
        dequant_targets = (
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
        )

    pass_instance = QuantizeFusedConvBnBiasPass(exported_program)
    result = pass_instance.call(exported_program.graph_module)

    assert result.modified

    # Every convolution bias should now flow through a dequantize node.
    for node in exported_program.graph_module.graph.nodes:
        if node.target not in conv_targets:
            continue
        bias = node.args[2]
        assert bias is not None, "Bias should not be None after pass"
        assert (
            bias.target in dequant_targets
        ), f"Bias should be dequantized, got {bias.target}"


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(ConvBnNoBias(), id="conv2d_bn_no_bias"),
        pytest.param(ConvBnReluNoBias(), id="conv2d_bn_relu_no_bias"),
    ],
)
def test_quantize_fused_convbn_bias_nxp_qat(model: nn.Module) -> None:
    """
    Test that QuantizeFusedConvBnBiasPass correctly quantizes the bias
    introduced by BatchNorm fusion during QAT when the original conv has bias=False.
    Uses the NXP Neutron quantizer with edge dialect.
    """
    _run_nxp_qat_pass(model, use_edge=True)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(ConvBnNoBias(), id="conv2d_bn_no_bias"),
        pytest.param(ConvBnReluNoBias(), id="conv2d_bn_relu_no_bias"),
    ],
)
def test_quantize_fused_convbn_bias_nxp_qat_aten(model: nn.Module) -> None:
    """
    Test that QuantizeFusedConvBnBiasPass correctly quantizes the bias
    on aten-dialect graphs (without edge conversion).
    Uses the NXP Neutron quantizer.
    """
    _run_nxp_qat_pass(model, use_edge=False)
