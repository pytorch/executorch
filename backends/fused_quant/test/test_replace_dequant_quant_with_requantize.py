# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
    get_zero_point,
)
from executorch.backends.fused_quant.optimization_passes.replace_dequant_quant_with_requantize import (
    ReplaceDequantQuantWithRequantize,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram
from torch.export.graph_signature import TensorArgument


class ReplaceDequantQuantWithRequantizeTest(unittest.TestCase):
    def _build_dequant_quant(
        self,
        *,
        include_float_output: bool = False,
        include_second_quant: bool = False,
    ) -> tuple[ExportedProgram, torch.Tensor]:
        builder = ProgramBuilder()
        inp = torch.tensor([[-4, 0, 7]], dtype=torch.int8)
        inp_node = builder.placeholder("inp", inp)
        dequant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(inp_node,),
            kwargs={
                "scale": 0.5,
                "zero_point": 1,
                "quant_min": -128,
                "quant_max": 127,
                "dtype": torch.int8,
            },
        )
        quant = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(dequant,),
            kwargs={
                "scale": 0.25,
                "zero_point": 3,
                "quant_min": 0,
                "quant_max": 255,
                "dtype": torch.uint8,
            },
        )
        outputs = [quant]
        if include_second_quant:
            outputs.append(
                builder.call_operator(
                    op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                    args=(dequant,),
                    kwargs={
                        "scale": 0.125,
                        "zero_point": 5,
                        "quant_min": 0,
                        "quant_max": 255,
                        "dtype": torch.uint8,
                    },
                )
            )
        if include_float_output:
            outputs.append(dequant)
        builder.output(outputs)
        return builder.get_program(), inp

    def test_replaces_dequant_quant(self) -> None:
        ep, inp = self._build_dequant_quant()
        expected = ep.module()(inp)

        result = ReplaceDequantQuantWithRequantize()(ep)

        self.assertTrue(result.modified)
        actual = result.exported_program.module()(inp)
        self.assertEqual(len(actual), len(expected))
        for actual_output, expected_output in zip(actual, expected):
            self.assertTrue(torch.equal(actual_output, expected_output))

        graph = result.exported_program.graph_module.graph
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                )
            ),
            0,
        )
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
            0,
        )
        (requantize,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.requantize.default
        )
        inp_qparams = get_qparams_from_node(requantize, "inp")
        out_qparams = get_qparams_from_node(requantize, "out")
        assert inp_qparams is not None and out_qparams is not None
        self.assertAlmostEqual(get_scale(result.exported_program, inp_qparams), 0.5)
        self.assertEqual(get_zero_point(result.exported_program, inp_qparams), 1)
        self.assertEqual(inp_qparams.dtype, torch.float32)
        self.assertAlmostEqual(get_scale(result.exported_program, out_qparams), 0.25)
        self.assertEqual(get_zero_point(result.exported_program, out_qparams), 3)
        self.assertEqual(out_qparams.dtype, torch.uint8)
        self.assertEqual(requantize.meta["val"].dtype, torch.uint8)
        self.assertEqual(
            result.exported_program.graph_signature.user_outputs,
            (requantize.name,),
        )
        result.exported_program.validate()

    def test_replaces_quant_branch_and_preserves_float_user(self) -> None:
        ep, inp = self._build_dequant_quant(include_float_output=True)
        expected = ep.module()(inp)

        result = ReplaceDequantQuantWithRequantize()(ep)

        self.assertTrue(result.modified)
        actual = result.exported_program.module()(inp)
        for actual_output, expected_output in zip(actual, expected):
            self.assertTrue(torch.equal(actual_output, expected_output))
        graph = result.exported_program.graph_module.graph
        (dequant,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
            0,
        )
        (requantize,) = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.requantize.default
        )
        self.assertEqual(
            result.exported_program.graph_signature.user_outputs,
            (requantize.name, dequant.name),
        )
        result.exported_program.validate()

    def test_replaces_multiple_quant_branches(self) -> None:
        ep, inp = self._build_dequant_quant(include_second_quant=True)
        expected = ep.module()(inp)

        result = ReplaceDequantQuantWithRequantize()(ep)

        self.assertTrue(result.modified)
        actual = result.exported_program.module()(inp)
        for actual_output, expected_output in zip(actual, expected):
            self.assertTrue(torch.equal(actual_output, expected_output))
        graph = result.exported_program.graph_module.graph
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                )
            ),
            0,
        )
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
            0,
        )
        requantize_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.requantize.default
        )
        self.assertEqual(len(requantize_nodes), 2)
        actual_out_qparams = []
        for requantize in requantize_nodes:
            out_qparams = get_qparams_from_node(requantize, "out")
            assert out_qparams is not None
            actual_out_qparams.append(
                (
                    get_scale(result.exported_program, out_qparams),
                    get_zero_point(result.exported_program, out_qparams),
                )
            )
        self.assertCountEqual(actual_out_qparams, [(0.25, 3), (0.125, 5)])
        first_inp_qparams = get_qparams_from_node(requantize_nodes[0], "inp")
        second_inp_qparams = get_qparams_from_node(requantize_nodes[1], "inp")
        assert first_inp_qparams is not None and second_inp_qparams is not None
        self.assertIs(first_inp_qparams.scale, second_inp_qparams.scale)
        self.assertIs(first_inp_qparams.zero_point, second_inp_qparams.zero_point)
        self.assertEqual(
            result.exported_program.graph_signature.user_outputs,
            tuple(requantize.name for requantize in requantize_nodes),
        )
        result.exported_program.validate()

    def test_replaces_multiple_quant_branches_and_preserves_float_user(self) -> None:
        ep, inp = self._build_dequant_quant(
            include_float_output=True,
            include_second_quant=True,
        )
        expected = ep.module()(inp)

        result = ReplaceDequantQuantWithRequantize()(ep)

        self.assertTrue(result.modified)
        actual = result.exported_program.module()(inp)
        for actual_output, expected_output in zip(actual, expected):
            self.assertTrue(torch.equal(actual_output, expected_output))
        graph = result.exported_program.graph_module.graph
        (dequant,) = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                )
            ),
            0,
        )
        requantize_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.requantize.default
        )
        self.assertEqual(len(requantize_nodes), 2)
        self.assertEqual(
            result.exported_program.graph_signature.user_outputs,
            (
                requantize_nodes[0].name,
                requantize_nodes[1].name,
                dequant.name,
            ),
        )
        result.exported_program.validate()

    def test_dequant_without_quant_user_is_noop(self) -> None:
        ep, _ = self._build_dequant_quant(include_float_output=True)
        graph = ep.graph_module.graph
        quant = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        )[0]
        quant.replace_all_uses_with(quant.args[0])
        graph.erase_node(quant)
        dequant = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        )[0]
        ep.graph_signature.output_specs[0].arg = TensorArgument(dequant.name)
        ep.graph_module.recompile()
        ep.validate()

        result = ReplaceDequantQuantWithRequantize()(ep)

        self.assertFalse(result.modified)
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.fused_quant.requantize.default,
                )
            ),
            0,
        )
