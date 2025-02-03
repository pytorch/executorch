# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
)
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester, RunPasses


class Sigmoid(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return x.sigmoid()

    def get_inputs(self):
        return (torch.rand(4),)


class TestInsertTablePass(unittest.TestCase):

    def test_insert_table_tosa_BI(self):
        module = Sigmoid()
        test_pass_stage = RunPasses(
            [FoldAndAnnotateQParamsPass],
            passes_with_exported_program=[InsertTableOpsPass],
        )
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .to_edge()
            .run_passes(test_pass_stage)
            .check("tosa._table")
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 1,
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
                }
            )
            .check_not(["aten_sigmoid_default"])
        )
