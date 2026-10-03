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
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.backends.fused_quant.optimization_passes.replace_scalar_mul_with_dequant_quant import (
    ReplaceScalarMulWithDequantQuant,
)
from executorch.backends.fused_quant.test.helpers import (
    create_per_axis_qparams,
    create_per_tensor_qparams,
)
from torch.export import ExportedProgram


_NULL_QPARAMS = (None, None, torch.float32, 0, 0)


class ReplaceScalarMulWithDequantQuantTest(unittest.TestCase):
    def _build_scalar_mul(
        self,
        *,
        scalar: float = 1.5,
        has_input_qparams: bool = True,
        has_output_qparams: bool = True,
        per_axis_input_qparams: bool = False,
    ) -> tuple[ExportedProgram, torch.Tensor]:
        builder = ProgramBuilder()
        inp_value = (
            torch.tensor([[-8, -1, 3, 8]], dtype=torch.int8)
            if has_input_qparams
            else torch.tensor([[-2.0, -0.25, 0.75, 2.0]])
        )
        inp = builder.placeholder("inp", inp_value)
        if per_axis_input_qparams:
            inp_qparams = create_per_axis_qparams(
                builder, num_channels=4, ndim=2, axis=1, dtype=torch.float32
            )
        elif has_input_qparams:
            inp_qparams = create_per_tensor_qparams(
                builder, scale=0.25, zero_point=-3, dtype=torch.float32
            )
        else:
            inp_qparams = _NULL_QPARAMS
        out_qparams = (
            create_per_tensor_qparams(
                builder, scale=0.2, zero_point=4, dtype=torch.int8
            )
            if has_output_qparams
            else _NULL_QPARAMS
        )
        mul = builder.call_operator(
            op=exir_ops.edge.fused_quant.mul.Scalar,
            args=(inp, *inp_qparams, *out_qparams, scalar),
        )
        builder.output([mul])
        return builder.get_program(), inp_value

    def _assert_numerics(
        self,
        ep: ExportedProgram,
        inp: torch.Tensor,
        expected: tuple[torch.Tensor, ...],
    ) -> None:
        actual = ep.module()(inp)
        self.assertEqual(len(expected), len(actual))
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(actual_output, expected_output)

    def test_replaces_fully_quantized_scalar_mul(self) -> None:
        ep, inp = self._build_scalar_mul()
        expected = ep.module()(inp)

        result = ReplaceScalarMulWithDequantQuant()(ep)

        self.assertTrue(result.modified)
        self._assert_numerics(result.exported_program, inp, expected)
        graph = result.exported_program.graph
        self.assertEqual(
            graph.find_nodes(
                op="call_function", target=exir_ops.edge.fused_quant.mul.Scalar
            ),
            [],
        )
        (dequant,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        (quant,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertLess(
            list(graph.nodes).index(dequant), list(graph.nodes).index(quant)
        )
        self.assertAlmostEqual(get_arg(dequant, "scale", float), 0.375)
        self.assertEqual(get_arg(dequant, "zero_point", int), -3)
        self.assertAlmostEqual(get_arg(quant, "scale", float), 0.2)
        self.assertEqual(get_arg(quant, "zero_point", int), 4)
        self.assertEqual(dequant.meta["val"].dtype, torch.float32)
        self.assertEqual(quant.meta["val"].dtype, torch.int8)
        result.exported_program.validate()

    def test_does_not_replace_without_both_qparam_blocks(self) -> None:
        for has_input_qparams, has_output_qparams in (
            (True, False),
            (False, True),
            (False, False),
        ):
            with self.subTest(
                has_input_qparams=has_input_qparams,
                has_output_qparams=has_output_qparams,
            ):
                ep, _ = self._build_scalar_mul(
                    has_input_qparams=has_input_qparams,
                    has_output_qparams=has_output_qparams,
                )

                result = ReplaceScalarMulWithDequantQuant()(ep)

                self.assertFalse(result.modified)
                self.assertEqual(
                    len(
                        result.exported_program.graph.find_nodes(
                            op="call_function",
                            target=exir_ops.edge.fused_quant.mul.Scalar,
                        )
                    ),
                    1,
                )

    def test_does_not_replace_unsupported_scalars(self) -> None:
        for scalar in (0.0, -1.5, float("inf"), float("nan")):
            with self.subTest(scalar=scalar):
                ep, _ = self._build_scalar_mul(scalar=scalar)

                result = ReplaceScalarMulWithDequantQuant()(ep)

                self.assertFalse(result.modified)

    def test_does_not_replace_per_axis_qparams(self) -> None:
        ep, _ = self._build_scalar_mul(per_axis_input_qparams=True)

        result = ReplaceScalarMulWithDequantQuant()(ep)

        self.assertFalse(result.modified)
