# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from numbers import Number
from typing import Tuple, Union

import pytest
import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.xnnpack.test.tester.tester import Quantize
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


test_data_suite = [
    # (test_name, test_data, min, max)
    ("rank_1", torch.rand(10) * 2, -1.0, 1.0),
    ("rank_2", torch.rand(1, 35), 0.5, 0.8),
    ("rank_3", torch.ones(1, 10, 10), -1, -1),
    ("rank_4", torch.rand(1, 10, 10, 1) * 2, -0.1, 2.0),
    ("rank_4_mixed_min_max_dtype", torch.rand(1, 10, 10, 5) + 10, 8.0, 10),
    ("rank_4_no_min", torch.rand(1, 10, 10, 1) * 10, None, 5),
    ("rank_4_no_max", torch.rand(1, 10, 10, 1) - 3, -3.3, None),
]


class TestClamp(unittest.TestCase):
    """Tests Clamp Operator."""

    class Clamp(torch.nn.Module):
        def __init__(
            self,
            min: Union[torch.Tensor, Number, None],
            max: Union[torch.Tensor, Number, None],
        ):
            super().__init__()

            self.clamp_min = min
            self.clamp_max = max

        def forward(self, x):
            return torch.clamp(x, self.clamp_min, self.clamp_max)

    def _test_clamp_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check(["torch.ops.aten.clamp.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_clamp_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        tosa_spec = TosaSpecification.create_from_string("TOSA-0.80+BI")
        compile_spec = common.get_tosa_compile_spec(tosa_spec)
        quantizer = ArmQuantizer(tosa_spec).set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.clamp.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_clamp_tosa_ethos_BI_pipeline(
        self,
        compile_spec: list[CompileSpec],
        module: torch.nn.Module,
        test_data: Tuple[torch.tensor],
    ):
        tosa_spec = TosaSpecification.create_from_compilespecs(compile_spec)
        quantizer = ArmQuantizer(tosa_spec).set_io(get_symmetric_quantization_config())
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.clamp.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_clamp_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        min: Union[torch.Tensor, Number, None],
        max: Union[torch.Tensor, Number, None],
    ):
        self._test_clamp_tosa_MI_pipeline(self.Clamp(min, max), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_clamp_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        min: Union[torch.Tensor, Number, None],
        max: Union[torch.Tensor, Number, None],
    ):
        self._test_clamp_tosa_BI_pipeline(self.Clamp(min, max), (test_data,))

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_clamp_tosa_u55_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        min: Union[torch.Tensor, Number, None],
        max: Union[torch.Tensor, Number, None],
    ):
        self._test_clamp_tosa_ethos_BI_pipeline(
            common.get_u55_compile_spec(), self.Clamp(min, max), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_clamp_tosa_u85_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        min: Union[torch.Tensor, Number, None],
        max: Union[torch.Tensor, Number, None],
    ):
        self._test_clamp_tosa_ethos_BI_pipeline(
            common.get_u85_compile_spec(), self.Clamp(min, max), (test_data,)
        )
