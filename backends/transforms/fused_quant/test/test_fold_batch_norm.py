# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import torch
from executorch.backends.transforms.fused_quant.pre_quantize_passes.fold_batch_norm import (
    FoldBatchNorm,
)
from executorch.backends.transforms.fused_quant.test.helpers import trace
from parameterized import parameterized
from torch import fx, nn
from torch._ops import OpOverload

_BATCH_NORM = torch.ops.aten._native_batch_norm_legit_no_training.default
_CONVOLUTION = torch.ops.aten.convolution.default


def _conv_target(input_rank: int) -> OpOverload:
    if input_rank == 3:
        return torch.ops.aten.conv1d.default
    if input_rank == 4:
        return torch.ops.aten.conv2d.default
    if input_rank == 5:
        return torch.ops.aten.conv3d.default
    raise ValueError(f"Unsupported input rank: {input_rank}")


def _make_conv(input_rank: int, channels: int, bias: bool) -> nn.Module:
    if input_rank == 3:
        return nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=bias)
    if input_rank == 4:
        return nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
    if input_rank == 5:
        return nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=bias)
    raise ValueError(f"Unsupported input rank: {input_rank}")


def _make_batch_norm(input_rank: int, channels: int) -> nn.Module:
    if input_rank == 3:
        return nn.BatchNorm1d(channels)
    if input_rank == 4:
        return nn.BatchNorm2d(channels)
    if input_rank == 5:
        return nn.BatchNorm3d(channels)
    raise ValueError(f"Unsupported input rank: {input_rank}")


def _randomize_batch_norm(batch_norm: nn.Module, channels: int) -> None:
    batch_norm.running_mean = torch.randn(channels)
    batch_norm.running_var = torch.rand(channels) + 0.5
    batch_norm.weight = nn.Parameter(torch.randn(channels))
    batch_norm.bias = nn.Parameter(torch.randn(channels))


class _ConvBatchNormModel(nn.Module):
    def __init__(self, input_rank: int, channels: int, bias: bool) -> None:
        super().__init__()
        self.conv = _make_conv(input_rank, channels, bias)
        self.batch_norm = _make_batch_norm(input_rank, channels)
        _randomize_batch_norm(self.batch_norm, channels)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(self.conv(inp))


def _trace(
    model: nn.Module, inp: torch.Tensor, conv_target: OpOverload
) -> fx.GraphModule:
    return trace(model, (inp,), ops_to_keep=[conv_target, _CONVOLUTION]).module()


def _count(graph_module: fx.GraphModule, target: object) -> int:
    return len(graph_module.graph.find_nodes(op="call_function", target=target))


class FoldBatchNormTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("conv1d_with_bias", (2, 3, 8), True),
            ("conv1d_without_bias", (2, 3, 8), False),
            ("conv2d_with_bias", (2, 3, 6, 8), True),
            ("conv2d_without_bias", (2, 3, 6, 8), False),
            ("conv3d_with_bias", (2, 3, 4, 6, 8), True),
            ("conv3d_without_bias", (2, 3, 4, 6, 8), False),
        ]
    )
    def test_folds_batch_norm_into_preceding_convolution(
        self,
        _name: str,
        input_shape: tuple[int, ...],
        bias: bool,
    ) -> None:
        torch.manual_seed(0)
        conv_target = _conv_target(len(input_shape))
        model = _ConvBatchNormModel(len(input_shape), input_shape[1], bias).eval()
        inp = torch.randn(input_shape)
        expected = model(inp)
        graph_module = _trace(model, inp, conv_target)

        self.assertEqual(_count(graph_module, conv_target), 1)
        self.assertEqual(_count(graph_module, _BATCH_NORM), 1)

        result = FoldBatchNorm()(graph_module)

        self.assertTrue(result.modified)
        self.assertEqual(_count(graph_module, conv_target), 1)
        self.assertEqual(_count(graph_module, _BATCH_NORM), 0)
        folded_conv = graph_module.graph.find_nodes(
            op="call_function", target=conv_target
        )[0]
        self.assertIsInstance(folded_conv.args[2], fx.Node)
        self.assertFalse(
            any(isinstance(value, fx.Node) for value in folded_conv.kwargs.values())
        )
        # The batch norm's statistics and the convolution's original weights are
        # gone; only the folded weights remain.
        weight, bias = folded_conv.args[1:3]
        assert isinstance(weight, fx.Node) and isinstance(bias, fx.Node)
        self.assertEqual(
            {weight.target, bias.target},
            {name for name, _ in graph_module.named_parameters()},
        )
        buffers = dict(graph_module.named_buffers())
        self.assertNotIn("batch_norm.running_mean", buffers)
        self.assertNotIn("batch_norm.running_var", buffers)
        actual = graph_module(inp)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
