# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester


class Add(torch.nn.Module):

    def get_inputs(self):
        return (torch.rand(1, 10, 10, 10),)

    def forward(self, x):
        return x + x


class TestTagIOQuantPass(unittest.TestCase):
    """Tests the TagIOQuantPass which tags q/dq nodes on model inputs and outputs to not include them in our partitions."""

    def _tosa_BI_u55_pipeline(self, module: torch.nn.Module):
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_u55_compile_spec(quantize_io=True),
            )
            .quantize()
            .export()
            .to_edge()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2
                }
            )
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2
                }
            )
            .partition()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 1
                }
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 1
                }
            )
            # .to_executorch() requires additional steps
        )

    def test_BI_u55_artifact(self):
        model = Add()
        self._tosa_BI_u55_pipeline(model)
