# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree..

import sys
from typing import Tuple
from unittest.mock import MagicMock

# Stub modules that are transitively imported by arm_quantizer but never
# exercised by these tests.
for _mod in ("tosa_serializer", "tosa", "tosa.TosaGraph"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ModuleNotFoundError:
            sys.modules[_mod] = MagicMock()

import pytest
import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.transforms.quantize_fused_convbn_bias_pass import (
    QuantizeFusedConvBnBiasAtenPass,
)
from torch import nn
from torch.export import export
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e


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


class ConvNoBnNoBias(nn.Module):
    """Conv2d with bias=False and no BatchNorm. Pass should skip this."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 3, 32, 32),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# --- Shared helpers ---


def _qat_prepare_convert(model, per_channel):
    """QAT prepare -> calibrate -> convert_pt2e, returns GraphModule with get_attr nodes."""
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(
        get_symmetric_quantization_config(is_qat=True, is_per_channel=per_channel)
    )
    example_input = model.get_inputs()
    exported = export(model, example_input, strict=True).module()
    prepared = prepare_qat_pt2e(exported, quantizer)
    prepared(*example_input)
    move_exported_model_to_eval(prepared)
    converted = convert_pt2e(prepared)
    return converted


def _assert_bias_dequantized(graph, conv_targets, dequant_targets):
    """Assert every conv's bias flows through a dequantize node."""
    conv_count = 0
    for node in graph.nodes:
        if node.target not in conv_targets:
            continue
        conv_count += 1
        bias = node.args[2]
        assert bias is not None, "Bias should not be None after pass"
        assert (
            bias.target in dequant_targets
        ), f"Bias should be dequantized, got {bias.target}"
    assert conv_count > 0, "Expected at least one convolution node"


# --- Direct aten pass tests ---

_aten_conv_targets = (
    torch.ops.aten.convolution.default,
    torch.ops.aten.conv2d.default,
)
_aten_dequant_targets = (
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
)

_aten_direct_models = [
    pytest.param((ConvBnNoBias, True), id="conv2d_bn_per_channel"),
    pytest.param((ConvBnNoBias, False), id="conv2d_bn_per_tensor"),
    pytest.param((ConvBnReluNoBias, True), id="conv2d_bn_relu_per_channel"),
    pytest.param((ConvBnReluNoBias, False), id="conv2d_bn_relu_per_tensor"),
]


@pytest.mark.parametrize("test_data", _aten_direct_models)
def test_aten_pass_direct(test_data) -> None:
    """QuantizeFusedConvBnBiasAtenPass on GraphModule (get_attr nodes, no EP)."""
    model_cls, per_channel = test_data
    gm = _qat_prepare_convert(model_cls(), per_channel)
    QuantizeFusedConvBnBiasAtenPass()(gm)
    _assert_bias_dequantized(gm.graph, _aten_conv_targets, _aten_dequant_targets)


@pytest.mark.parametrize("test_data", _aten_direct_models)
def test_aten_pass_with_exported_program(test_data) -> None:
    """QuantizeFusedConvBnBiasAtenPass on graph_module from EP (placeholder nodes)."""
    model_cls, per_channel = test_data
    model = model_cls()
    gm = _qat_prepare_convert(model, per_channel)
    ep = export(gm, model.get_inputs(), strict=True)
    QuantizeFusedConvBnBiasAtenPass(ep)(ep.graph_module)
    _assert_bias_dequantized(
        ep.graph_module.graph, _aten_conv_targets, _aten_dequant_targets
    )


def test_aten_pass_idempotent() -> None:
    """Running the pass twice doesn't break."""
    model = ConvBnNoBias()
    gm = _qat_prepare_convert(model, per_channel=True)
    QuantizeFusedConvBnBiasAtenPass()(gm)
    QuantizeFusedConvBnBiasAtenPass()(gm)
    _assert_bias_dequantized(gm.graph, _aten_conv_targets, _aten_dequant_targets)


def test_aten_pass_biasless_conv_no_bn() -> None:
    """Pass skips biasless conv without following BatchNorm."""
    model = ConvNoBnNoBias()
    gm = _qat_prepare_convert(model, per_channel=True)
    QuantizeFusedConvBnBiasAtenPass()(gm)
