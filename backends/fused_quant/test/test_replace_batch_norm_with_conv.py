# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from typing import cast

import torch
from executorch.backends.cadence.aot.compiler import trace
from executorch.backends.fused_quant.graph_utils import get_constant
from executorch.backends.fused_quant.pre_quantize_passes.replace_batch_norm_with_conv import (
    ReplaceBatchNormWithConv,
)
from parameterized import parameterized
from torch import fx, nn
from torch.export import ExportedProgram

_NATIVE_BATCH_NORM = torch.ops.aten._native_batch_norm_legit_no_training.default
_CONVOLUTION = torch.ops.aten.convolution.default


class _BatchNormModel(nn.Module):
    def __init__(
        self,
        channels: int,
        input_rank: int,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if input_rank in (2, 3):
            self.batch_norm = nn.BatchNorm1d(channels, affine=affine)
        elif input_rank == 4:
            self.batch_norm = nn.BatchNorm2d(channels, affine=affine)
        elif input_rank == 5:
            self.batch_norm = nn.BatchNorm3d(channels, affine=affine)
        else:
            raise ValueError(f"Unsupported input rank: {input_rank}")

        self.batch_norm.running_mean = torch.randn(channels)
        self.batch_norm.running_var = torch.rand(channels) + 0.5
        if affine:
            self.batch_norm.weight = nn.Parameter(torch.randn(channels))
            self.batch_norm.bias = nn.Parameter(torch.randn(channels))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(inp)


class _NativeAuxOutputModel(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels))
        self.bias = nn.Parameter(torch.randn(channels))
        self.register_buffer("running_mean", torch.randn(channels))
        self.register_buffer("running_var", torch.rand(channels) + 0.5)

    def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, saved_mean, _saved_invstd = _NATIVE_BATCH_NORM(
            inp,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            0.1,
            1e-5,
        )
        return output, saved_mean


def _trace(model: nn.Module, inp: torch.Tensor) -> ExportedProgram:
    return trace(model, (inp,))


def _only(exported_program: ExportedProgram, target: object) -> fx.Node:
    nodes = exported_program.graph_module.graph.find_nodes(
        op="call_function", target=target
    )
    assert len(nodes) == 1
    return nodes[0]


class ReplaceBatchNormWithConvTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("batch_norm_1d_without_length", (2, 3)),
            ("batch_norm_1d", (2, 3, 5)),
            ("batch_norm_2d", (2, 3, 4, 5)),
            ("batch_norm_3d", (2, 3, 2, 4, 5)),
        ]
    )
    def test_replaces_with_dimension_generic_depthwise_convolution(
        self,
        _name: str,
        input_shape: tuple[int, ...],
    ) -> None:
        torch.manual_seed(0)
        model = _BatchNormModel(input_shape[1], len(input_shape)).eval()
        inp = torch.randn(input_shape)
        expected = model(inp)
        exported_program = _trace(model, inp)

        self.assertEqual(
            len(
                exported_program.graph_module.graph.find_nodes(
                    op="call_function", target=_NATIVE_BATCH_NORM
                )
            ),
            1,
        )

        result = ReplaceBatchNormWithConv()(exported_program)

        self.assertTrue(result.modified)
        exported_program = result.exported_program
        self.assertEqual(
            len(
                exported_program.graph_module.graph.find_nodes(
                    op="call_function", target=_NATIVE_BATCH_NORM
                )
            ),
            0,
        )
        conv = _only(exported_program, _CONVOLUTION)
        weight = get_constant(exported_program, cast(fx.Node, conv.args[1]))
        self.assertIsNotNone(weight)
        spatial_dims = max(len(input_shape) - 2, 1)
        self.assertEqual(
            tuple(cast(torch.Tensor, weight).shape),
            (input_shape[1], 1, *([1] * spatial_dims)),
        )
        self.assertEqual(conv.args[3], [1] * spatial_dims)
        self.assertEqual(conv.args[4], [0] * spatial_dims)
        self.assertEqual(conv.args[5], [1] * spatial_dims)
        self.assertFalse(conv.args[6])
        self.assertEqual(conv.args[7], [0] * spatial_dims)
        self.assertEqual(conv.args[8], input_shape[1])

        actual = exported_program.module()(inp)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_supports_batch_norm_without_affine_parameters(self) -> None:
        torch.manual_seed(0)
        model = _BatchNormModel(3, input_rank=4, affine=False).eval()
        inp = torch.randn(2, 3, 4, 5)
        expected = model(inp)
        exported_program = _trace(model, inp)

        result = ReplaceBatchNormWithConv()(exported_program)

        self.assertTrue(result.modified)
        actual = result.exported_program.module()(inp)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_preserves_source_metadata_on_convolution(self) -> None:
        model = _BatchNormModel(3, input_rank=4).eval()
        inp = torch.randn(2, 3, 4, 5)
        exported_program = _trace(model, inp)
        sentinel = [("batch_norm", torch.nn.functional.batch_norm)]
        _only(exported_program, _NATIVE_BATCH_NORM).meta["source_fn_stack"] = sentinel

        result = ReplaceBatchNormWithConv()(exported_program)

        self.assertEqual(
            _only(result.exported_program, _CONVOLUTION).meta["source_fn_stack"],
            sentinel,
        )

    def test_native_batch_norm_with_used_auxiliary_output_is_not_replaced(self) -> None:
        model = _NativeAuxOutputModel(3).eval()
        inp = torch.randn(2, 3, 4, 5)
        exported_program = _trace(model, inp)

        result = ReplaceBatchNormWithConv()(exported_program)

        self.assertFalse(result.modified)
        self.assertEqual(
            len(
                result.exported_program.graph_module.graph.find_nodes(
                    op="call_function", target=_NATIVE_BATCH_NORM
                )
            ),
            1,
        )
