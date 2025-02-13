# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.backends.xnnpack.test.tester.tester import Quantize
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized


test_data_suite = [
    # (test_name, test_data, [kernel_size, stride, padding])
    ("zeros", torch.zeros(1, 16, 50, 32), [4, 2, 0]),
    ("ones", torch.zeros(1, 16, 50, 32), [4, 2, 0]),
    ("rand", torch.rand(1, 16, 50, 32), [4, 2, 0]),
    ("randn", torch.randn(1, 16, 50, 32), [4, 2, 0]),
]


class TestAvgPool2d(unittest.TestCase):
    """Tests AvgPool2d."""

    class AvgPool2d(torch.nn.Module):
        def __init__(
            self,
            kernel_size: int | Tuple[int, int],
            stride: int | Tuple[int, int],
            padding: int | Tuple[int, int],
        ):
            super().__init__()
            self.avg_pool_2d = torch.nn.AvgPool2d(
                kernel_size=kernel_size, stride=stride, padding=padding
            )

        def forward(self, x):
            return self.avg_pool_2d(x)

    def _test_avgpool2d_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check(["torch.ops.aten.avg_pool2d.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_avgpool2d_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        tosa_spec = TosaSpecification.create_from_string("TOSA-0.80+BI")
        compile_spec = common.get_tosa_compile_spec(tosa_spec)
        quantizer = TOSAQuantizer(tosa_spec).set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.avg_pool2d.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_avgpool2d_tosa_ethos_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.tensor],
    ):
        quantizer = EthosUQuantizer(compile_spec).set_io(
            get_symmetric_quantization_config()
        )
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.avg_pool2d.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_avgpool2d_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params: int | Tuple[int, int],
    ):
        self._test_avgpool2d_tosa_MI_pipeline(
            self.AvgPool2d(*model_params), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    def test_avgpool2d_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params: int | Tuple[int, int],
    ):
        self._test_avgpool2d_tosa_BI_pipeline(
            self.AvgPool2d(*model_params), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_avgpool2d_tosa_u55_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params: int | Tuple[int, int],
    ):
        self._test_avgpool2d_tosa_ethos_BI_pipeline(
            self.AvgPool2d(*model_params),
            common.get_u55_compile_spec(),
            (test_data,),
        )

    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    def test_avgpool2d_tosa_u85_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params: int | Tuple[int, int],
    ):
        self._test_avgpool2d_tosa_ethos_BI_pipeline(
            self.AvgPool2d(*model_params),
            common.get_u85_compile_spec(),
            (test_data,),
        )

    reject_data_suite = [
        (AvgPool2d(1, 1, 0), torch.rand(2, 5, 5, 5)),
        (AvgPool2d((2, 9), 1, 1), torch.rand(1, 16, 5, 32)),
        (AvgPool2d(1, 4, 0), torch.rand(1, 10, 10, 10)),
        (AvgPool2d((1, 257), 1, 0), torch.rand(1, 16, 5, 300)),
        (AvgPool2d((800, 90), 1, 0), torch.rand(1, 16, 850, 100)),
    ]

    @parameterized.expand(reject_data_suite)
    def test_reject_avgpool2d_u55_BI(
        self,
        module: torch.nn.Module,
        test_data: torch.tensor,
    ):
        compile_spec = common.get_u55_compile_spec()
        quantizer = EthosUQuantizer(compile_spec).set_io(
            get_symmetric_quantization_config()
        )

        (
            ArmTester(
                module,
                example_inputs=(test_data,),
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.avg_pool2d.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check(["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 0})
        )
