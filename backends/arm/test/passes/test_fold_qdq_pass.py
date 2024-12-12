# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from executorch.backends.xnnpack.test.tester.tester import RunPasses

from executorch.exir.dialects._ops import ops as exir_ops


class SimpleQuantizeModel(torch.nn.Module):
    def forward(self, x, y):
        return x + torch.max((x + x), (y + y))

    def get_inputs(self):
        return (torch.rand(1, 1280, 7, 7), torch.rand(1, 1280, 7, 7))


class FoldAndAnnotateQParamsPassTestClass(FoldAndAnnotateQParamsPass):
    def __init__(self):
        super(FoldAndAnnotateQParamsPassTestClass, self).__init__(
            [
                exir_ops.edge.aten.add.Tensor,
                exir_ops.edge.aten.maximum.default,
            ]
        )


class TestFoldAndAnnotateQParamsPass(unittest.TestCase):
    """
    Tests the FoldAndAnnotateQParamsPass which folds dq/q nodes into
    the node and stores the quantization parameters in meta.
    """

    def test_fold_qdq_pass(self):
        """
        Check that the pass runs for add operation and that one q node and one dq node
        is removed from the representation.
        """
        module = SimpleQuantizeModel()
        test_pass_stage = RunPasses([FoldAndAnnotateQParamsPassTestClass])
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .to_edge()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 7,
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 6,
                }
            )
            .run_passes(test_pass_stage)
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1,
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
                }
            )
        )
