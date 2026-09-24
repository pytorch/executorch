# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa: F401
import executorch.backends.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.fused_quant.test.helpers import (
    create_per_axis_qparams,
    create_per_channel_qparams,
    create_per_tensor_qparams,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops


class ToChannelsFirstTest(unittest.TestCase):
    def assert_bit_exact(self, expected: object, actual: object) -> None:
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_convolution(self) -> None:
        inp = torch.randint(-8, 8, (1, 6, 6, 2), dtype=torch.int8)
        weight = torch.randint(-8, 8, (3, 3, 3, 2), dtype=torch.int8)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        weight_node = builder.placeholder("weight", weight)
        inp_qparams = create_per_axis_qparams(
            builder, 2, 4, axis=3, dtype=torch.float32, vary_values=True
        )
        weight_qparams = create_per_channel_qparams(
            builder, 3, dtype=torch.float32, weight_ndim=4
        )
        out_qparams = create_per_axis_qparams(builder, 3, 4, axis=3, vary_values=True)
        convolution = builder.call_operator(
            exir_ops.edge.fused_quant.convolution_channels_last.default,
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
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
            ),
        )
        builder.output([convolution])
        program = builder.get_program()
        expected = program.module()(inp, weight)

        result = ToChannelsFirst().call(program.graph_module)

        self.assertTrue(result.modified)
        self.assert_bit_exact(expected, program.module()(inp, weight))
        self.assertEqual(
            result.graph_module.graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.fused_quant.convolution_channels_last.default,
            ),
            [],
        )
        [channels_first] = result.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.convolution.default,
        )
        self.assertFalse(get_arg(channels_first, "transposed", bool))
        self.assertEqual(get_arg(channels_first, "output_padding", list[int]), [0, 0])

    def test_max_pool2d_with_indices(self) -> None:
        inp = torch.randint(-8, 8, (1, 6, 6, 2), dtype=torch.int8)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        inp_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32)
        out_qparams = create_per_tensor_qparams(builder, scale=0.5)
        pool = builder.call_operator(
            exir_ops.edge.fused_quant.max_pool2d_with_indices_channels_last.default,
            args=(
                inp_node,
                *inp_qparams,
                *out_qparams,
                [3, 3],
                [2, 2],
                [1, 1],
                [1, 1],
                False,
            ),
        )
        values = builder.call_getitem(pool, 0)
        indices = builder.call_getitem(pool, 1)
        builder.output([values, indices])
        program = builder.get_program()
        expected = program.module()(inp)

        result = ToChannelsFirst().call(program.graph_module)

        self.assertTrue(result.modified)
        self.assert_bit_exact(expected, program.module()(inp))
        self.assertEqual(
            len(
                result.graph_module.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.fused_quant.max_pool2d_with_indices.default,
                )
            ),
            1,
        )

    def test_avg_pool2d(self) -> None:
        inp = torch.randint(-8, 8, (1, 6, 6, 2), dtype=torch.int8)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        inp_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32)
        out_qparams = create_per_tensor_qparams(builder, scale=0.25)
        pool = builder.call_operator(
            exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
            args=(
                inp_node,
                *inp_qparams,
                *out_qparams,
                [3, 3],
                [2, 2],
                [1, 1],
                False,
                True,
                None,
            ),
        )
        builder.output([pool])
        program = builder.get_program()
        expected = program.module()(inp)

        result = ToChannelsFirst().call(program.graph_module)

        self.assertTrue(result.modified)
        self.assert_bit_exact(expected, program.module()(inp))
        self.assertEqual(
            len(
                result.graph_module.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.fused_quant.avg_pool2d.default,
                )
            ),
            1,
        )
