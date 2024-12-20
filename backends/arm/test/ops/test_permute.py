# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import pytest

import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.xnnpack.test.tester.tester import Quantize
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized
from torchvision.ops import Permute

test_data_suite = [
    # (test_name,test_data,dims)
    ("rank_2", torch.rand(10, 10), [1, 0]),
    ("rank_3", torch.rand(10, 10, 10), [2, 0, 1]),
    ("rank_3", torch.rand(10, 10, 10), [1, 2, 0]),
    ("rank_4", torch.rand(1, 5, 1, 10), [0, 2, 3, 1]),
    ("rank_4", torch.rand(1, 2, 5, 10), [1, 0, 2, 3]),
    ("rank_4", torch.rand(1, 10, 10, 5), [2, 0, 1, 3]),
]


class TestPermute(unittest.TestCase):
    """Tests Permute Operator."""

    class Permute(torch.nn.Module):

        def __init__(self, dims: list[int]):
            super().__init__()

            self.permute = Permute(dims=dims)

        def forward(self, x):
            return self.permute(x)

    def _test_permute_tosa_MI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.tensor],
        permute_memory_to_nhwc: bool,
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+MI", permute_memory_to_nhwc=permute_memory_to_nhwc
                ),
            )
            .export()
            .check(["torch.ops.aten.permute.default"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_permute_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.tensor]
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.permute.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_permute_ethos_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.Tensor],
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.permute.default": 1})
            .check(["torch.ops.quantized_decomposed"])
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_permute_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_permute_tosa_MI(
        self, test_name: str, test_data: torch.Tensor, dims: list[int]
    ):
        self._test_permute_tosa_MI_pipeline(self.Permute(dims=dims), (test_data,), True)
        self._test_permute_tosa_MI_pipeline(
            self.Permute(dims=dims), (test_data,), False
        )

    @parameterized.expand(test_data_suite)
    def test_permute_tosa_BI(
        self, test_name: str, test_data: torch.Tensor, dims: list[int]
    ):
        self._test_permute_tosa_BI_pipeline(self.Permute(dims=dims), (test_data,))

    # Expected to fail as TOSA.Transpose is not supported by Ethos-U55.
    @parameterized.expand(test_data_suite[0:1])
    @pytest.mark.corstone_fvp
    @unittest.expectedFailure
    def test_permute_u55_BI(
        self, test_name: str, test_data: torch.Tensor, dims: list[int]
    ):
        self._test_permute_ethos_BI_pipeline(
            self.Permute(dims=dims), common.get_u55_compile_spec(), (test_data,)
        )

    @parameterized.expand(test_data_suite[:-2])
    @pytest.mark.corstone_fvp
    def test_permute_u85_BI(
        self, test_name: str, test_data: torch.Tensor, dims: list[int]
    ):
        self._test_permute_ethos_BI_pipeline(
            self.Permute(dims=dims), common.get_u85_compile_spec(), (test_data,)
        )

    # Fails since on FVP since N > 1 is not supported. MLETORCH-517
    @parameterized.expand(test_data_suite[-2:])
    @pytest.mark.corstone_fvp
    @conftest.expectedFailureOnFVP
    def test_permute_u85_BI_xfails(
        self, test_name: str, test_data: torch.Tensor, dims: list[int]
    ):
        self._test_permute_ethos_BI_pipeline(
            self.Permute(dims=dims), common.get_u85_compile_spec(), (test_data,)
        )
