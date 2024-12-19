# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized


class TestMeanDim(unittest.TestCase):
    """Tests MeanDim, called AdaptiveAvgPool2d in Pytorch."""

    class AdaptiveAveragePool2d(torch.nn.Module):
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

        def __init__(self):
            super().__init__()
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        def forward(self, x):
            return self.adaptive_avg_pool2d(x)

    class MeanDim(torch.nn.Module):
        test_data_suite = [
            # (test_name, test_data)
            ("zeros", torch.zeros(1, 1280, 7, 7), -1, True),
            ("ones", torch.ones(1, 1280, 7, 7), (-1, 2), False),
            (
                "rand",
                torch.rand(1, 1280, 7, 7),
                (-1),
                True,
            ),
            (
                "randn",
                torch.randn(1, 1280, 7, 7),
                (-1, -2, -3),
                False,
            ),
        ]

        def __init__(self, dim: int | list[int] = -1, keepdim: bool = True):
            super().__init__()
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x: torch.Tensor):
            return x.mean(dim=self.dim, keepdim=self.keepdim)

    def _test_adaptive_avg_pool2d_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
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

    def _test_adaptive_avg_pool2d_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
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
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_adaptive_avg_pool2d_tosa_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .check(["torch.ops.aten.adaptive_avg_pool2d.default"])
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

    def _test_meandim_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
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
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1.0)
        )

    def _test_meandim_tosa_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
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

    @parameterized.expand(AdaptiveAveragePool2d.test_data_suite)
    def test_adaptive_avg_pool2d_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_adaptive_avg_pool2d_tosa_MI_pipeline(
            self.AdaptiveAveragePool2d(), (test_data,)
        )

    @parameterized.expand(AdaptiveAveragePool2d.test_data_suite)
    def test_adaptive_avg_pool2d_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_adaptive_avg_pool2d_tosa_BI_pipeline(
            self.AdaptiveAveragePool2d(), (test_data,)
        )

    @parameterized.expand(AdaptiveAveragePool2d.test_data_suite)
    def test_adaptive_avg_pool2d_tosa_u55_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_adaptive_avg_pool2d_tosa_ethosu_BI_pipeline(
            self.AdaptiveAveragePool2d(), common.get_u55_compile_spec(), (test_data,)
        )

    @parameterized.expand(AdaptiveAveragePool2d.test_data_suite)
    def test_adaptive_avg_pool2d_tosa_u85_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
    ):
        self._test_adaptive_avg_pool2d_tosa_ethosu_BI_pipeline(
            self.AdaptiveAveragePool2d(), common.get_u85_compile_spec(), (test_data,)
        )

    @parameterized.expand(MeanDim.test_data_suite)
    def test_meandim_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int | list[int] = -1,
        keepdim: bool = True,
    ):
        self._test_meandim_tosa_MI_pipeline(self.MeanDim(dim, keepdim), (test_data,))

    @parameterized.expand(MeanDim.test_data_suite)
    def test_meandim_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int | list[int] = -1,
        keepdim: bool = True,
    ):
        self._test_meandim_tosa_BI_pipeline(self.MeanDim(dim, keepdim), (test_data,))

    # Expected to fail as this is not supported on u55.
    @parameterized.expand(MeanDim.test_data_suite)
    @unittest.expectedFailure
    def test_meandim_tosa_u55_BI_xfails(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int | list[int] = -1,
        keepdim: bool = True,
    ):
        self._test_meandim_tosa_ethosu_BI_pipeline(
            self.MeanDim(dim, keepdim),
            common.get_u55_compile_spec(),
            (test_data,),
        )

    @parameterized.expand(MeanDim.test_data_suite)
    def test_meandim_tosa_u85_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        dim: int | list[int] = -1,
        keepdim: bool = True,
    ):
        self._test_meandim_tosa_ethosu_BI_pipeline(
            self.MeanDim(dim, keepdim),
            common.get_u85_compile_spec(),
            (test_data,),
        )
