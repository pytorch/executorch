# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs


class SimpleModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y

    def get_inputs(self):
        a = torch.rand(1, 2, 2, 1)
        b = torch.rand(1, 2, 2, 1)
        return (a, b)


class TestIOQuantizationPass(unittest.TestCase):
    """
    Test the executorch/exir/passes/quanize_io_pass pass works(meaning we don't get Q/DQ nodes) on a simple model
    """

    def test_ioquantisation_pass(self):
        model = SimpleModel()
        tester = (
            ArmTester(
                model,
                example_inputs=model.get_inputs(),
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3
                }
            )
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3
                }
            )
            .partition()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2
                }
            )
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1
                }
            )
        )
        edge = tester.get_artifact()
        edge.transform(
            passes=[QuantizeInputs(edge, [0, 1]), QuantizeOutputs(edge, [0])]
        )
        tester.check_not(["edge__ops_quantized_decomposed_quantize_per_tensor"])
        tester.check_not(["edge__ops_quantized_decomposed_dequantize_per_tensor"])
