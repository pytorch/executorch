# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import torch
from executorch.backends.cadence.aot.compiler import trace
from executorch.backends.fused_quant.pre_quantize_passes.fold_batch_norm import (
    FoldBatchNorm,
)
from executorch.backends.fused_quant.pre_quantize_passes.replace_batch_norm_with_conv import (
    ReplaceBatchNormWithConv,
)
from parameterized import parameterized
from torch import fx, nn
from torch._ops import OpOverload
from torch.export import ExportedProgram

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


class _FoldAndReplaceModel(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fusable_batch_norm = nn.BatchNorm2d(channels)
        self.standalone_batch_norm = nn.BatchNorm2d(channels)
        _randomize_batch_norm(self.fusable_batch_norm, channels)
        _randomize_batch_norm(self.standalone_batch_norm, channels)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.fusable_batch_norm(self.conv(inp)) + self.standalone_batch_norm(inp)


def _trace(
    model: nn.Module, inp: torch.Tensor, conv_target: OpOverload
) -> ExportedProgram:
    return trace(model, (inp,), ops_to_keep=[conv_target, _CONVOLUTION])


def _count(exported_program: ExportedProgram, target: object) -> int:
    return len(
        exported_program.graph_module.graph.find_nodes(
            op="call_function", target=target
        )
    )


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
        exported_program = _trace(model, inp, conv_target)

        self.assertEqual(_count(exported_program, conv_target), 1)
        self.assertEqual(_count(exported_program, _BATCH_NORM), 1)

        result = FoldBatchNorm()(exported_program)

        self.assertTrue(result.modified)
        self.assertEqual(_count(result.exported_program, conv_target), 1)
        self.assertEqual(_count(result.exported_program, _BATCH_NORM), 0)
        folded_conv = result.exported_program.graph_module.graph.find_nodes(
            op="call_function", target=conv_target
        )[0]
        self.assertIsInstance(folded_conv.args[2], fx.Node)
        self.assertFalse(
            any(isinstance(value, fx.Node) for value in folded_conv.kwargs.values())
        )
        actual = result.exported_program.module()(inp)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_fold_then_replace_handles_only_remaining_batch_norm(self) -> None:
        torch.manual_seed(0)
        model = _FoldAndReplaceModel(3).eval()
        inp = torch.randn(2, 3, 6, 8)
        expected = model(inp)
        exported_program = _trace(model, inp, torch.ops.aten.conv2d.default)

        fold_result = FoldBatchNorm()(exported_program)

        self.assertTrue(fold_result.modified)
        self.assertEqual(_count(fold_result.exported_program, _BATCH_NORM), 1)

        replace_result = ReplaceBatchNormWithConv()(fold_result.exported_program)

        self.assertTrue(replace_result.modified)
        self.assertEqual(_count(replace_result.exported_program, _BATCH_NORM), 0)
        self.assertEqual(_count(replace_result.exported_program, _CONVOLUTION), 1)
        actual = replace_result.exported_program.module()(inp)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
