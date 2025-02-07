# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the expand op which copies the data of the input tensor (possibly with new data format)
#

import unittest

from typing import Sequence, Tuple

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
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized


class TestSimpleExpand(unittest.TestCase):
    """Tests the Tensor.expand which should be converted to a repeat op by a pass."""

    class Expand(torch.nn.Module):
        # (input tensor, multiples)
        test_parameters = [
            (torch.rand(1), (2,)),
            (torch.randn(1, 4), (1, -1)),
            (torch.randn(1), (2, 2, 4)),
            (torch.randn(1, 1, 1, 5), (1, 4, -1, -1)),
            (torch.randn(1, 1, 192), (1, -1, -1)),
            (torch.randn(1, 1), (1, 2, 2, 4)),
            (torch.randn(1, 1), (2, 2, 2, 4)),
            (torch.randn(10, 1, 1, 97), (-1, 4, -1, -1)),
            (torch.rand(1, 1, 2, 2), (4, 3, -1, 2)),
        ]

        def forward(self, x: torch.Tensor, m: Sequence):
            return x.expand(m)

    def _test_expand_tosa_MI_pipeline(self, module: torch.nn.Module, test_data: Tuple):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.expand.default": 1})
            .to_edge()
            .partition()
            .check_not(["torch.ops.aten.expand.default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_expand_tosa_BI_pipeline(self, module: torch.nn.Module, test_data: Tuple):
        tosa_spec = TosaSpecification.create_from_string("TOSA-0.80+BI")
        compile_spec = common.get_tosa_compile_spec(tosa_spec)
        quantizer = ArmQuantizer(tosa_spec).set_io(get_symmetric_quantization_config())
        (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check_count({"torch.ops.aten.expand.default": 1})
            .to_edge()
            .partition()
            .check_not(["torch.ops.aten.expand.default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_expand_ethosu_BI_pipeline(
        self, compile_spec: CompileSpec, module: torch.nn.Module, test_data: Tuple
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
            .check_count({"torch.ops.aten.expand.default": 1})
            .to_edge()
            .partition()
            .check_not(["torch.ops.aten.expand.default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(Expand.test_parameters)
    def test_expand_tosa_MI(self, test_input, multiples):
        self._test_expand_tosa_MI_pipeline(self.Expand(), (test_input, multiples))

    @parameterized.expand(Expand.test_parameters)
    def test_expand_tosa_BI(self, test_input, multiples):
        self._test_expand_tosa_BI_pipeline(self.Expand(), (test_input, multiples))

    @parameterized.expand(Expand.test_parameters[:-3])
    @pytest.mark.corstone_fvp
    def test_expand_u55_BI(self, test_input, multiples):
        self._test_expand_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Expand(), (test_input, multiples)
        )

    # MLETORCH-629: Expand does not work on FVP with batch>1
    @parameterized.expand(Expand.test_parameters[-3:])
    @pytest.mark.corstone_fvp
    @conftest.expectedFailureOnFVP
    def test_expand_u55_BI_xfails(self, test_input, multiples):
        self._test_expand_ethosu_BI_pipeline(
            common.get_u55_compile_spec(), self.Expand(), (test_input, multiples)
        )

    @parameterized.expand(Expand.test_parameters[:-3])
    @pytest.mark.corstone_fvp
    def test_expand_u85_BI(self, test_input, multiples):
        self._test_expand_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Expand(), (test_input, multiples)
        )

    # MLETORCH-629: Expand does not work on FVP with batch>1
    @parameterized.expand(Expand.test_parameters[-3:])
    @pytest.mark.corstone_fvp
    @conftest.expectedFailureOnFVP
    def test_expand_u85_BI_xfails(self, test_input, multiples):
        self._test_expand_ethosu_BI_pipeline(
            common.get_u85_compile_spec(), self.Expand(), (test_input, multiples)
        )
