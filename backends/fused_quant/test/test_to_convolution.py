# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from typing import Any

import executorch.backends.cadence.aot.ops_registrations  # noqa: F401
import executorch.backends.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.fused_quant.optimization_passes.to_convolution import (
    ToConvolution,
)
from executorch.backends.fused_quant.test.helpers import create_per_tensor_qparams
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized


class ToConvolutionTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "conv1d",
                exir_ops.edge.fused_quant.conv1d.default,
                (1, 2, 8),
                (3, 2, 3),
                [2],
                [1],
                [1],
            ),
            (
                "conv2d",
                exir_ops.edge.fused_quant.conv2d.default,
                (1, 2, 8, 8),
                (3, 2, 3, 3),
                [2, 2],
                [1, 1],
                [1, 1],
            ),
            (
                "conv3d",
                exir_ops.edge.fused_quant.conv3d.default,
                (1, 2, 8, 8, 8),
                (3, 2, 3, 3, 3),
                [2, 2, 2],
                [1, 1, 1],
                [1, 1, 1],
            ),
        ]
    )
    def test_converts_to_convolution(
        self,
        _name: str,
        target: Any,
        input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        stride: list[int],
        padding: list[int],
        dilation: list[int],
    ) -> None:
        inp = torch.randint(-8, 8, input_shape, dtype=torch.int8)
        weight = torch.randint(-8, 8, weight_shape, dtype=torch.int8)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        weight_node = builder.placeholder("weight", weight)
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        weight_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        out_qparams = create_per_tensor_qparams(builder)
        conv = builder.call_operator(
            target,
            args=(
                inp_node,
                weight_node,
                None,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.float32,
                -128,
                127,
                *out_qparams,
                stride,
                padding,
                dilation,
                1,
            ),
        )
        builder.output([conv])
        program = builder.get_program()
        expected = program.module()(inp, weight)

        result = ToConvolution().call(program.graph_module)

        self.assertTrue(result.modified)
        torch.testing.assert_close(
            expected, program.module()(inp, weight), rtol=0, atol=0
        )
        self.assertEqual(
            result.graph_module.graph.find_nodes(op="call_function", target=target),
            [],
        )
        [convolution] = result.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.convolution.default,
        )
        self.assertFalse(get_arg(convolution, "transposed", bool))
        self.assertEqual(
            get_arg(convolution, "output_padding", list[int]), [0] * len(stride)
        )
