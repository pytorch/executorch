# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir import EdgeCompileConfig
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


test_data_suite_rank1 = [
    # (test_name, test_data, out_features, has_bias)
    (
        "model_linear_rank1_zeros",
        torch.zeros(10),
        15,
        True,
    ),
    (
        "model_linear_rank1_ones",
        torch.ones(10),
        15,
        False,
    ),
    (
        "model_linear_rank1_negative_ones",
        torch.ones(10) * (-1),
        20,
        True,
    ),
    (
        "model_linear_rank1_rand",
        torch.rand(10),
        10,
        True,
    ),
    (
        "model_linear_rank1_negative_large_rand",
        torch.rand(10) * (-100),
        30,
        False,
    ),
    (
        "model_linear_rank1_large_randn",
        torch.randn(15) * 100,
        20,
        True,
    ),
]

test_data_suite_rank4 = [
    # (test_name, test_data, out_features, has_bias)
    (
        "model_linear_rank4_zeros",
        torch.zeros(5, 10, 25, 20),
        30,
        True,
    ),
    (
        "model_linear_rank4_ones",
        torch.ones(5, 10, 25, 20),
        30,
        False,
    ),
    (
        "model_linear_rank4_negative_ones",
        torch.ones(5, 10, 25, 20) * (-1),
        30,
        True,
    ),
    (
        "model_linear_rank4_rand",
        torch.rand(5, 10, 25, 20),
        30,
        False,
    ),
    (
        "model_linear_rank4_negative_large_rand",
        torch.rand(5, 10, 25, 20) * (-100),
        30,
        True,
    ),
    (
        "model_linear_rank4_large_randn",
        torch.randn(5, 10, 25, 20) * 100,
        30,
        False,
    ),
]


class TestLinear(unittest.TestCase):
    """tests the linear operation y = Ax + b"""

    _edge_compile_config: EdgeCompileConfig = EdgeCompileConfig(
        _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
    )

    class Linear(torch.nn.Module):
        def __init__(
            self,
            in_features: int,
            out_features: int = 3,
            bias: bool = True,
        ):
            super().__init__()
            self.fc = torch.nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
            )

        def forward(self, x):
            return self.fc(x)

    def _test_linear_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+MI", permute_memory_to_nhwc=True
                ),
            )
            .export()
            .check_count({"torch.ops.aten.linear.default": 1})
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower(edge_compile_config=self._edge_compile_config)
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_linear_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+BI", permute_memory_to_nhwc=True
                ),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.linear.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower(edge_compile_config=self._edge_compile_config)
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_linear_tosa_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.Tensor],
    ) -> ArmTester:
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.linear.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower(edge_compile_config=self._edge_compile_config)
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        return tester

    @parameterized.expand(test_data_suite_rank1 + test_data_suite_rank4)
    def test_linear_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        out_features: int,
        has_bias: bool,
    ):
        in_features = test_data.shape[-1]
        test_data = (test_data,)
        self._test_linear_tosa_MI_pipeline(
            self.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
            ),
            test_data,
        )

    @parameterized.expand(test_data_suite_rank1 + test_data_suite_rank4)
    def test_linear_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        out_features: int,
        has_bias: bool,
    ):
        in_features = test_data.shape[-1]
        test_data = (test_data,)
        self._test_linear_tosa_BI_pipeline(
            self.Linear(
                in_features=in_features, out_features=out_features, bias=has_bias
            ),
            test_data,
        )

    @parameterized.expand(test_data_suite_rank1)
    @pytest.mark.corstone_fvp
    def test_linear_tosa_u55_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        out_features: int,
        has_bias: bool,
    ):
        in_features = test_data.shape[-1]
        test_data = (test_data,)
        tester = self._test_linear_tosa_ethosu_BI_pipeline(
            self.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
            ),
            common.get_u55_compile_spec(),
            test_data,
        )

        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite_rank1 + test_data_suite_rank4)
    def test_linear_tosa_u85_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        out_features: int,
        has_bias: bool,
    ):
        in_features = test_data.shape[-1]
        test_data = (test_data,)
        self._test_linear_tosa_ethosu_BI_pipeline(
            self.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=has_bias,
            ),
            common.get_u85_compile_spec(),
            test_data,
        )
