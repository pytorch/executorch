# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.backends.cadence.aot.compiler import (
    apply_pre_edge_transform_passes,
    get_fake_quant_model,
)
from executorch.backends.cadence.aot.fold_qat_conv_bn import FoldQATConvBNPass
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceDefaultQuantizer
from executorch.backends.transforms.quantize_fused_convbn_bias_pass import (
    QuantizeFusedConvBnBiasAtenPass,
)
from torch import nn
from torch.export import export


class Conv1dBnModel(nn.Module):
    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv1d(3, 8, kernel_size=3, padding=1, bias=bias)
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseConv1dBnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(6, 6, kernel_size=3, padding=1, groups=6, bias=False)
        self.bn = nn.BatchNorm1d(6)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class MultiConvBnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 8, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(8, 16, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.bn3(self.conv3(x))


def _count_op(graph: torch.fx.Graph, target_substr: str) -> int:
    return sum(1 for n in graph.nodes if target_substr in str(n.target))


def _get_qat_graphmodule(
    model: nn.Module, example_input: tuple[torch.Tensor, ...]
) -> torch.fx.GraphModule:
    quantizer = CadenceDefaultQuantizer()
    return get_fake_quant_model(model, example_input, quantizer)


class FoldQATConvBNTest(unittest.TestCase):
    def test_conv1d_bn_with_bias(self) -> None:
        model = Conv1dBnModel(bias=True)
        example_input = (torch.randn(1, 3, 32),)
        gm = _get_qat_graphmodule(model, example_input)
        self.assertEqual(
            _count_op(gm.graph, "batch_norm"),
            0,
        )

    def test_conv1d_bn_no_bias(self) -> None:
        model = Conv1dBnModel(bias=False)
        example_input = (torch.randn(1, 3, 32),)
        gm = _get_qat_graphmodule(model, example_input)
        self.assertEqual(
            _count_op(gm.graph, "batch_norm"),
            0,
        )

    def test_depthwise_conv1d_bn_no_bias(self) -> None:
        model = DepthwiseConv1dBnModel()
        example_input = (torch.randn(1, 6, 32),)
        gm = _get_qat_graphmodule(model, example_input)
        self.assertEqual(
            _count_op(gm.graph, "batch_norm"),
            0,
        )

    def test_multi_conv_bn(self) -> None:
        model = MultiConvBnModel()
        example_input = (torch.randn(1, 3, 32),)
        gm = _get_qat_graphmodule(model, example_input)
        self.assertEqual(
            _count_op(gm.graph, "batch_norm"),
            0,
        )

    def test_simulation_chain_removed(self) -> None:
        model = Conv1dBnModel(bias=True)
        example_input = (torch.randn(1, 3, 32),)
        gm = _get_qat_graphmodule(model, example_input)
        self.assertEqual(_count_op(gm.graph, "aten.sqrt"), 0)
        self.assertEqual(_count_op(gm.graph, "batch_norm"), 0)

    def test_on_exported_program(self) -> None:
        model = MultiConvBnModel()
        model.eval()
        example_input = (torch.randn(1, 3, 32),)
        quantizer = CadenceDefaultQuantizer()
        gm = get_fake_quant_model(model, example_input, quantizer)
        ep = export(gm, example_input, strict=True)
        ep = apply_pre_edge_transform_passes(ep, quantizer)
        self.assertEqual(
            _count_op(ep.graph_module.graph, "batch_norm"),
            0,
        )

    def test_pass_is_idempotent(self) -> None:
        model = Conv1dBnModel(bias=True)
        example_input = (torch.randn(1, 3, 32),)
        quantizer = CadenceDefaultQuantizer()
        gm = get_fake_quant_model(model, example_input, quantizer)
        QuantizeFusedConvBnBiasAtenPass(default_zero_bias=True)(gm)
        FoldQATConvBNPass()(gm)
        FoldQATConvBNPass()(gm)
        self.assertEqual(_count_op(gm.graph, "batch_norm"), 0)

    def test_no_bn_model_unchanged(self) -> None:
        class NoBnModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(3, 8, kernel_size=3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.conv(x))

        model = NoBnModel()
        example_input = (torch.randn(1, 3, 32),)
        gm = _get_qat_graphmodule(model, example_input)
        node_count_before = len(list(gm.graph.nodes))
        QuantizeFusedConvBnBiasAtenPass(default_zero_bias=True)(gm)
        FoldQATConvBNPass()(gm)
        node_count_after = len(list(gm.graph.nodes))
        self.assertEqual(node_count_before, node_count_after)
