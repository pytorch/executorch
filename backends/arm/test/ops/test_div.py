# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from typing import Optional, Tuple, Union

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

test_data_suite = [
    # (test_name, input, other, rounding_mode) See torch.div() for info
    (
        "op_div_rank1_ones",
        torch.ones(5),
        torch.ones(5),
        None,
    ),
    (
        "op_div_rank1_negative_ones",
        torch.ones(5) * (-1),
        torch.ones(5) * (-1),
        None,
    ),
    (
        "op_div_rank1_rand",
        torch.rand(5) * 5,
        torch.rand(5) * 5,
        None,
    ),
    (
        "op_div_rank4_ones",
        torch.ones(5, 10, 25, 20),
        torch.ones(5, 10, 25, 20),
        None,
    ),
    (
        "op_div_rank4_negative_ones",
        (-1) * torch.ones(5, 10, 25, 20),
        torch.ones(5, 10, 25, 20),
        None,
    ),
    (
        "op_div_rank4_ones_div_negative",
        torch.ones(5, 10, 25, 20),
        (-1) * torch.ones(5, 10, 25, 20),
        None,
    ),
    (
        "op_div_rank4_large_rand",
        200 * torch.rand(5, 10, 25, 20),
        torch.rand(5, 10, 25, 20),
        None,
    ),
    (
        "op_div_rank4_negative_large_rand",
        (-200) * torch.rand(5, 10, 25, 20),
        torch.rand(5, 10, 25, 20),
        None,
    ),
    (
        "op_div_rank4_large_randn",
        200 * torch.randn(5, 10, 25, 20) + 1,
        torch.rand(5, 10, 25, 20) + 1,
        None,
    ),
]


class TestDiv(unittest.TestCase):
    """Tests division"""

    class Div(torch.nn.Module):

        def forward(
            self,
            input_: Union[torch.Tensor, torch.types.Number],
            other_: Union[torch.Tensor, torch.types.Number],
            rounding_mode: Optional[str] = None,
        ):
            if rounding_mode is None:
                return torch.div(input=input_, other=other_)
            else:
                return torch.div(
                    input=input_, other=other_, rounding_mode=rounding_mode
                )

    def _test_div_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.div.Tensor": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_div_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .check_count(
                {"torch.ops.aten.reciprocal.default": 1, "torch.ops.aten.mul.Tensor": 1}
            )
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, atol=1, rtol=0.1)
        )

    def _test_div_ethos_BI_pipeline(
        self, module: torch.nn.Module, compile_spec, test_data: Tuple[torch.Tensor]
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count(
                {"torch.ops.aten.reciprocal.default": 1, "torch.ops.aten.mul.Tensor": 1}
            )
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_div_tosa_MI(
        self,
        test_name: str,
        input_: Union[torch.Tensor, torch.types.Number],
        other_: Union[torch.Tensor, torch.types.Number],
        rounding_mode: Optional[str] = None,
    ):
        test_data = (input_, other_)
        self._test_div_tosa_MI_pipeline(self.Div(), test_data)

    @parameterized.expand(test_data_suite)
    def test_div_tosa_BI(
        self,
        test_name: str,
        input_: Union[torch.Tensor, torch.types.Number],
        other_: Union[torch.Tensor, torch.types.Number],
        rounding_mode: Optional[str] = None,
    ):

        test_data = (input_, other_)
        self._test_div_tosa_BI_pipeline(self.Div(), test_data)

    @parameterized.expand(test_data_suite[:2])
    @pytest.mark.corstone_fvp
    def test_div_u55_BI(
        self,
        test_name: str,
        input_: Union[torch.Tensor, torch.types.Number],
        other_: Union[torch.Tensor, torch.types.Number],
        rounding_mode: Optional[str] = None,
    ):
        test_data = (input_, other_)
        self._test_div_ethos_BI_pipeline(
            self.Div(), common.get_u55_compile_spec(), test_data
        )

    # Numerical issues on FVP likely due to mul op, MLETORCH-521
    @parameterized.expand(test_data_suite[2:])
    @pytest.mark.corstone_fvp
    @conftest.expectedFailureOnFVP
    def test_div_u55_BI_xfails(
        self,
        test_name: str,
        input_: Union[torch.Tensor, torch.types.Number],
        other_: Union[torch.Tensor, torch.types.Number],
        rounding_mode: Optional[str] = None,
    ):
        test_data = (input_, other_)
        self._test_div_ethos_BI_pipeline(
            self.Div(), common.get_u55_compile_spec(), test_data
        )

    @parameterized.expand(test_data_suite[:2])
    @pytest.mark.corstone_fvp
    def test_div_u85_BI(
        self,
        test_name: str,
        input_: Union[torch.Tensor, torch.types.Number],
        other_: Union[torch.Tensor, torch.types.Number],
        rounding_mode: Optional[str] = None,
    ):
        test_data = (input_, other_)
        self._test_div_ethos_BI_pipeline(
            self.Div(), common.get_u85_compile_spec(), test_data
        )

    # Numerical issues on FVP likely due to mul op, MLETORCH-521
    @parameterized.expand(test_data_suite[2:])
    @pytest.mark.corstone_fvp
    @conftest.expectedFailureOnFVP
    def test_div_u85_BI_xfails(
        self,
        test_name: str,
        input_: Union[torch.Tensor, torch.types.Number],
        other_: Union[torch.Tensor, torch.types.Number],
        rounding_mode: Optional[str] = None,
    ):
        test_data = (input_, other_)
        self._test_div_ethos_BI_pipeline(
            self.Div(), common.get_u85_compile_spec(), test_data
        )
