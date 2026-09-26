# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa: F401
import executorch.backends.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.fused_quant.decompose_fused_quant import DecomposeFusedQuant
from executorch.backends.fused_quant.test.helpers import (
    create_per_axis_qparams,
    create_per_channel_qparams,
    create_per_tensor_qparams,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind


class DecomposeFusedQuantTest(unittest.TestCase):
    def assert_bit_exact(self, expected: object, actual: object) -> None:
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def _assert_no_fused_quant_ops(self, program: ExportedProgram) -> None:
        self.assertFalse(
            any(
                node.op == "call_function"
                and getattr(node.target, "namespace", None) == "fused_quant"
                for node in program.graph.nodes
            )
        )

    def test_decomposes_per_channel_convolution(self) -> None:
        inp = torch.randint(-8, 8, (1, 2, 6, 6), dtype=torch.int8)
        weight = torch.randint(-8, 8, (3, 2, 3, 3), dtype=torch.int8)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        weight_node = builder.placeholder("weight", weight)
        inp_qparams = create_per_axis_qparams(
            builder, 2, 4, axis=1, dtype=torch.float32, vary_values=True
        )
        weight_qparams = create_per_channel_qparams(
            builder, 3, dtype=torch.float32, weight_ndim=4
        )
        out_qparams = create_per_axis_qparams(builder, 3, 4, axis=1, vary_values=True)
        convolution = builder.call_operator(
            exir_ops.edge.fused_quant.convolution.default,
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

        decomposed = DecomposeFusedQuant()(program)

        self.assertTrue(decomposed.modified)
        self.assert_bit_exact(
            expected,
            decomposed.exported_program.module()(inp, weight),
        )
        self._assert_no_fused_quant_ops(decomposed.exported_program)
        self.assertEqual(
            len(
                decomposed.exported_program.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.aten.convolution.default,
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                decomposed.exported_program.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                decomposed.exported_program.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
                )
            ),
            1,
        )

    def test_preserves_transposed_convolution_arguments(self) -> None:
        inp = torch.randn(1, 2, 4, 4)
        weight = torch.randn(2, 3, 3, 3)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        weight_node = builder.placeholder("weight", weight)
        no_qparams = (None, None, torch.float32, 0, 0)
        convolution = builder.call_operator(
            exir_ops.edge.fused_quant.convolution.default,
            args=(
                inp_node,
                weight_node,
                None,
                *no_qparams,
                *no_qparams,
                *no_qparams,
                *no_qparams,
                [2, 2],
                [1, 1],
                [1, 1],
                True,
                [1, 1],
                1,
            ),
        )
        builder.output([convolution])
        program = builder.get_program()
        expected = program.module()(inp, weight)

        result = DecomposeFusedQuant()(program)

        self.assertTrue(result.modified)
        self.assert_bit_exact(expected, result.exported_program.module()(inp, weight))
        [aten_convolution] = result.exported_program.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.aten.convolution.default,
        )
        self.assertTrue(get_arg(aten_convolution, "transposed", bool))
        self.assertEqual(get_arg(aten_convolution, "output_padding", list[int]), [1, 1])

    def test_decomposes_tensor_add(self) -> None:
        inp = torch.tensor([[-4, 1, 7]], dtype=torch.int8)
        other = torch.tensor([[2, -3, 5]], dtype=torch.int8)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        other_node = builder.placeholder("other", other)
        inp_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32)
        other_qparams = create_per_tensor_qparams(
            builder, scale=0.25, dtype=torch.float32
        )
        out_qparams = create_per_tensor_qparams(builder, scale=0.125, zero_point=3)
        add = builder.call_operator(
            exir_ops.edge.fused_quant.add.default,
            args=(
                inp_node,
                other_node,
                *inp_qparams,
                *other_qparams,
                *out_qparams,
                1.0,
            ),
        )
        builder.output([add])
        program = builder.get_program()
        expected = program.module()(inp, other)

        result = DecomposeFusedQuant()(program)

        self.assertTrue(result.modified)
        self.assert_bit_exact(expected, result.exported_program.module()(inp, other))
        self._assert_no_fused_quant_ops(result.exported_program)
        graph = result.exported_program.graph
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.aten.add.Tensor,
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
            1,
        )

    def test_decomposes_requantize_without_aten_op(self) -> None:
        inp = torch.tensor([[-4, 1, 7]], dtype=torch.int8)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        inp_qparams = create_per_tensor_qparams(builder, scale=0.5, dtype=torch.float32)
        out_qparams = create_per_tensor_qparams(builder, scale=0.25, zero_point=2)
        requantize = builder.call_operator(
            exir_ops.edge.fused_quant.requantize.default,
            args=(inp_node, *inp_qparams, *out_qparams),
        )
        builder.output([requantize])
        program = builder.get_program()
        expected = program.module()(inp)

        result = DecomposeFusedQuant()(program)

        self.assertTrue(result.modified)
        self.assert_bit_exact(expected, result.exported_program.module()(inp))
        self._assert_no_fused_quant_ops(result.exported_program)
        call_targets = [
            node.target
            for node in result.exported_program.graph.nodes
            if node.op == "call_function" and node.target is not operator.getitem
        ]
        self.assertEqual(
            call_targets.count(
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ),
            1,
        )
        self.assertEqual(
            call_targets.count(
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            ),
            1,
        )

    def test_rejects_group_quantization(self) -> None:
        inp = torch.randint(-8, 8, (2, 4), dtype=torch.int8)
        builder = ProgramBuilder()
        inp_node = builder.placeholder("inp", inp)
        scale = builder.placeholder(
            "group_scale",
            torch.ones(1, 2, dtype=torch.float32),
            input_kind=InputKind.BUFFER,
        )
        zero_point = builder.placeholder(
            "group_zero_point",
            torch.zeros(1, 2, dtype=torch.int64),
            input_kind=InputKind.BUFFER,
        )
        relu = builder.call_operator(
            exir_ops.edge.fused_quant.relu.default,
            args=(
                inp_node,
                scale,
                zero_point,
                torch.float32,
                -128,
                127,
                None,
                None,
                torch.float32,
                -128,
                127,
            ),
        )
        builder.output([relu])
        program = builder.get_program()

        with self.assertRaisesRegex(
            ValueError,
            "only per-tensor and per-channel quantization are supported",
        ):
            DecomposeFusedQuant()(program)
