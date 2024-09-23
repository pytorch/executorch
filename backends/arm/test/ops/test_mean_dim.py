# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

test_data_suite = [
    # (test_name, test_data)
    (
        "zeros",
        torch.zeros(1, 1280, 7, 7),
    ),
    (
        "ones",
        torch.ones(1, 1280, 7, 7),
    ),
    (
        "rand",
        torch.rand(1, 1280, 7, 7),
    ),
    (
        "randn",
        torch.randn(1, 1280, 7, 7),
    ),
]


class TestMeanDim(unittest.TestCase):
    """Tests MeanDim, called AdaptiveAvgPool2d in Pytorch."""

    class MeanDim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean_dim = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        def forward(self, x):
            return self.mean_dim(x)

    def _test_meandim_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check(["torch.ops.aten.adaptive_avg_pool2d.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_meandim_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.adaptive_avg_pool2d.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_meandim_tosa_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.adaptive_avg_pool2d.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_mean_dim",
                    "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default",
                ]
            )
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(test_data_suite)
    def test_meandim_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_meandim_tosa_MI_pipeline(self.MeanDim(), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_meandim_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_meandim_tosa_BI_pipeline(self.MeanDim(), (test_data,))

    @parameterized.expand(test_data_suite)
    def test_meandim_tosa_u55_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_meandim_tosa_u55_BI_pipeline(self.MeanDim(), (test_data,))
