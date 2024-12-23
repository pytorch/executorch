# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.xnnpack.test.tester.tester import Quantize
from executorch.exir.backend.compile_spec_schema import CompileSpec


class TestModel(torch.nn.Module):

    def get_inputs(self):
        return (torch.rand(1, 10, 10, 10), (torch.rand(1, 10, 10, 10)))

    def forward(self, x, y):
        result = x + y
        result = result * y
        result = result * x
        result = result - y
        return result


class TestTagUnquantizedNodesPass(unittest.TestCase):
    """
    Tests the TagUnquantizedNodesPass which tags unquantized nodes on model
    to not include them in our partitions.
    """

    def _tosa_BI_pipeline(
        self, module: torch.nn.Module, compile_spec: list[CompileSpec]
    ):
        quantizer = ArmQuantizer()
        # Quantize only add and sub nodes
        quantizer.STATIC_ANNOTATION_ORDER = [
            "add",
            "sub",
        ]
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=compile_spec,
            )
            .quantize(
                Quantize(
                    quantizer,
                    get_symmetric_quantization_config(is_per_channel=False),
                )
            )
            .export()
            .to_edge()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 5
                }
            )
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 6
                }
            )
            .partition()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 3
                }
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 2})
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2
                }
            )
        )

    def test_BI_u55_artifact(self):
        model = TestModel()
        self._tosa_BI_pipeline(
            model,
            common.get_u55_compile_spec(
                quantize_io=True, unquantized_nodes_to_cpu=True
            ),
        )

    def test_BI_u85_artifact(self):
        model = TestModel()
        self._tosa_BI_pipeline(
            model,
            common.get_u85_compile_spec(
                quantize_io=True, unquantized_nodes_to_cpu=True
            ),
        )
