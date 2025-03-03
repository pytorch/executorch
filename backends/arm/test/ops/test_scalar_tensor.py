# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.xnnpack.test.tester.tester import Quantize
from parameterized import parameterized


float_test_data_suite = [
    # (test_name, scalar input, scalar input type,)
    (
        "scalar_tensor_float_1",
        3.7,
        torch.float32,
    ),
    (
        "scalar_tensor_float_2",
        66,
        torch.float32,
    ),
]

int_test_data_suite = [
    # (test_name, scalar input, scalar input type,)
    (
        "scalar_tensor_int32",
        33,
        torch.int32,
    ),
    (
        "scalar_tensor_int8",
        8,
        torch.int8,
    ),
    (
        "scalar_tensor_int16",
        16 * 16 * 16,
        torch.int16,
    ),
]


class ScalarTensor(torch.nn.Module):
    def __init__(self, scalar, dtype=torch.float32):
        super().__init__()
        self.scalar = scalar
        self.dtype = dtype

    def forward(self):
        return torch.scalar_tensor(self.scalar, dtype=self.dtype)


class TestScalarTensor(unittest.TestCase):

    def _test_scalar_tensor_tosa_MI_pipeline(
        self, module: torch.nn.Module, expected_output
    ):
        test_outputs = []
        in_data = ()

        (
            ArmTester(
                module,
                example_inputs=in_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+MI",
                ),
            )
            .export()
            .check_count({"torch.ops.aten.scalar_tensor.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_get_output(test_outputs, inputs=in_data)
        )
        self._verify_output(test_outputs, expected_output)

    def _test_scalar_tensor_tosa_BI_pipeline(
        self, module: torch.nn.Module, expected_output
    ):
        test_outputs = []
        in_data = ()
        tosa_spec = TosaSpecification.create_from_string("TOSA-0.80+BI")
        compile_spec = common.get_tosa_compile_spec(tosa_spec)
        quantizer = TOSAQuantizer(tosa_spec).set_io(get_symmetric_quantization_config())

        (
            ArmTester(
                module,
                example_inputs=in_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.full.default": 1})  # Already replaced
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_get_output(test_outputs, inputs=in_data)
        )
        self._verify_output(test_outputs, expected_output)

    def _verify_output(self, test_outputs, expected_output):
        out_data = torch.squeeze(test_outputs[0][0])
        assert out_data == expected_output
        assert out_data.dtype == expected_output.dtype

    @parameterized.expand(int_test_data_suite + float_test_data_suite)
    def test_scalar_tensor_tosa_MI(  # Note TOSA MI supports all types
        self, test_name: str, scalar_value, scalar_type
    ):
        scalar = scalar_value
        dtype = scalar_type
        self._test_scalar_tensor_tosa_MI_pipeline(
            ScalarTensor(scalar, dtype), torch.scalar_tensor(scalar, dtype=dtype)
        )

    @parameterized.expand(float_test_data_suite)
    def test_scalar_tensor_tosa_BI(self, test_name: str, scalar_value, scalar_type):
        scalar = scalar_value
        dtype = scalar_type
        self._test_scalar_tensor_tosa_BI_pipeline(
            ScalarTensor(scalar, dtype), torch.scalar_tensor(scalar, dtype=dtype)
        )
