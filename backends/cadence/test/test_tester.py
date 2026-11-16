# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.cadence.aot.pass_utils import CompileMode
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceDefaultQuantizer
from executorch.backends.cadence.runtime.runtime import get_op_names
from executorch.backends.cadence.test.tester import CadenceTester
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops


def _quantized_linear_tester(mode: CompileMode = CompileMode.DEFAULT) -> CadenceTester:
    return (
        CadenceTester(
            torch.nn.Linear(4, 4),
            (torch.randn(2, 4),),
            quantizer=CadenceDefaultQuantizer(),
            mode=mode,
        )
        .quantize()
        .export()
    )


def _call_targets(program: torch.export.ExportedProgram) -> list[object]:
    return [node.target for node in program.graph.nodes if node.op == "call_function"]


class CadenceTesterTest(unittest.TestCase):
    def test_lowering_leaves_the_reference_program_unchanged(self) -> None:
        tester = _quantized_linear_tester()
        reference_code = tester.get_artifact(StageType.EXPORT).graph_module.code

        tester.to_edge_transform_and_lower()

        self.assertEqual(
            tester.get_artifact(StageType.EXPORT).graph_module.code, reference_code
        )

    def test_lowers_to_cadence_kernels_and_records_the_edge_program(self) -> None:
        tester = _quantized_linear_tester().to_edge_transform_and_lower(
            generate_etrecord=True
        )

        self.assertEqual(
            _call_targets(tester.get_artifact()._etrecord.edge_dialect_program),
            [
                exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                exir_ops.edge.cadence.quantized_linear.per_tensor,
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            ],
        )

        tester.to_executorch()

        self.assertEqual(
            get_op_names(tester.get_artifact().executorch_program),
            {
                "cadence::quantize_per_tensor.out",
                "cadence::quantized_linear.per_tensor_out",
                "cadence::dequantize_per_tensor.out",
            },
        )

    def test_size_mode_reaches_the_cadence_passes(self) -> None:
        tester = (
            _quantized_linear_tester(CompileMode.SIZE)
            .to_edge_transform_and_lower()
            .to_executorch()
        )

        self.assertEqual(
            get_op_names(tester.get_artifact().executorch_program),
            {
                "cadence::quantize_per_tensor_asym8s.out",
                "cadence::quantized_linear_asym8sxasym8s_asym8s.per_tensor_out",
                "cadence::dequantize_per_tensor_asym8s.out",
            },
        )

    def test_lowers_a_float_program_without_a_quantizer(self) -> None:
        tester = (
            CadenceTester(torch.nn.Linear(4, 4), (torch.randn(2, 4),))
            .export()
            .to_edge_transform_and_lower(generate_etrecord=True)
        )

        self.assertEqual(
            _call_targets(tester.get_artifact()._etrecord.edge_dialect_program),
            [exir_ops.edge.aten.permute_copy.default, exir_ops.edge.aten.addmm.default],
        )

        tester.to_executorch()

        self.assertEqual(
            get_op_names(tester.get_artifact().executorch_program),
            {"aten::linear.out"},
        )
