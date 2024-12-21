# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the clone op which copies the data of the input tensor (possibly with new data format)
#

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


class TestSimpleClone(unittest.TestCase):
    """Tests clone."""

    class Clone(torch.nn.Module):
        sizes = [10, 15, 50, 100]
        test_parameters = [(torch.ones(n),) for n in sizes]

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            x = x.clone()
            return x

    def _test_clone_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: torch.Tensor
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.clone.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_clone_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
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
            .check_count({"torch.ops.aten.clone.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_clone_tosa_ethos_pipeline(
        self,
        compile_spec: list[CompileSpec],
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        tester = (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.clone.default": 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    def _test_clone_tosa_u55_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        self._test_clone_tosa_ethos_pipeline(
            common.get_u55_compile_spec(), module, test_data
        )

    def _test_clone_tosa_u85_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        self._test_clone_tosa_ethos_pipeline(
            common.get_u85_compile_spec(), module, test_data
        )

    @parameterized.expand(Clone.test_parameters)
    def test_clone_tosa_MI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_MI_pipeline(self.Clone(), (test_tensor,))

    @parameterized.expand(Clone.test_parameters)
    def test_clone_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_BI_pipeline(self.Clone(), (test_tensor,))

    @parameterized.expand(Clone.test_parameters)
    @pytest.mark.corstone_fvp
    def test_clone_u55_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_u55_pipeline(self.Clone(), (test_tensor,))

    @parameterized.expand(Clone.test_parameters)
    @pytest.mark.corstone_fvp
    def test_clone_u85_BI(self, test_tensor: torch.Tensor):
        self._test_clone_tosa_u85_pipeline(self.Clone(), (test_tensor,))
