# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm.arm_backend import generate_ethosu_compile_spec

from executorch.backends.arm.test.tester.arm_tester import ArmTester


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.inputs = (torch.ones(1, 1, 1, in_features),)
        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def get_inputs(self):
        return self.inputs

    def forward(self, x):
        return self.fc(x)


class TestTagIOQuantPass(unittest.TestCase):

    def _tosa_BI_u55_pipeline(self, module: torch.nn.Module):
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=generate_ethosu_compile_spec(
                    "ethos-u55-128",
                    permute_memory_to_nhwc=True,
                    quantize_io=True,
                ),
            )
            .quantize()
            .export()
            .to_edge()
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
        model = Linear(20, 30)
        self._tosa_BI_u55_pipeline(model)
