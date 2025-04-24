# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the mean op which changes the size of a Tensor without changing the underlying data.
#

import unittest

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


class TestVar(unittest.TestCase):

    class Var(torch.nn.Module):
        test_parameters = [
            (torch.randn(1, 50, 10, 20), True, 0),
            (torch.rand(1, 50, 10), False, 0),
            (torch.randn(1, 30, 15, 20), True, 1),
            (torch.rand(1, 50, 10, 20), False, 0.5),
        ]

        def __init__(self, keepdim: bool = True, correction: int = 0):
            super().__init__()
            self.keepdim = keepdim
            self.correction = correction

        def forward(
            self,
            x: torch.Tensor,
        ):
            return x.var(keepdim=self.keepdim, correction=self.correction)

    class VarDim(torch.nn.Module):
        test_parameters = [
            (torch.randn(1, 50, 10, 20), 1, True, False),
            (torch.rand(1, 50, 10), -2, False, False),
            (torch.randn(1, 30, 15, 20), -3, True, True),
            (torch.rand(1, 50, 10, 20), -1, False, True),
        ]

        test_parameters_u55 = [
            (torch.randn(1, 50, 10, 20), 1, True, False),
            (torch.randn(1, 30, 15, 20), -3, True, True),
        ]

        test_parameters_u55_xfails = [
            (torch.rand(1, 50, 10), -2, True, False),
            (torch.rand(1, 50, 10, 20), -1, True, True),
        ]

        def __init__(self, dim: int = -1, keepdim: bool = True, unbiased: bool = False):
            super().__init__()
            self.dim = dim
            self.keepdim = keepdim
            self.unbiased = unbiased

        def forward(
            self,
            x: torch.Tensor,
        ):
            return x.var(dim=self.dim, keepdim=self.keepdim, unbiased=self.unbiased)

    class VarCorrection(torch.nn.Module):
        test_parameters = [
            (torch.randn(1, 50, 10, 20), (-1, -2), True, 0),
            (torch.rand(1, 50, 10), (-2), True, 0),
            (torch.randn(1, 30, 15, 20), (-1, -2, -3), True, 1),
            (torch.rand(1, 50, 10, 20), (-1, -2), True, 0.5),
        ]

        def __init__(
            self, dim: int = -1, keepdim: bool = True, correction: bool = False
        ):
            super().__init__()
            self.dim = dim
            self.keepdim = keepdim
            self.correction = correction

        def forward(
            self,
            x: torch.Tensor,
        ):
            return x.var(dim=self.dim, keepdim=self.keepdim, correction=self.correction)

    def _test_var_tosa_MI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: torch.Tensor,
        target_str: str = None,
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_var_tosa_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: torch.Tensor,
        target_str: str = None,
    ):
        tosa_spec = TosaSpecification.create_from_string("TOSA-0.80+BI")
        compile_spec = common.get_tosa_compile_spec(tosa_spec)
        quantizer = TOSAQuantizer(tosa_spec).set_io(get_symmetric_quantization_config())
        (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_var_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: torch.Tensor,
        target_str: str = None,
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
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(inputs=test_data, qtol=1)

    @parameterized.expand(Var.test_parameters)
    def test_var_tosa_MI(self, test_tensor: torch.Tensor, keepdim, correction):
        self._test_var_tosa_MI_pipeline(self.Var(keepdim, correction), (test_tensor,))

    @parameterized.expand(Var.test_parameters)
    def test_var_tosa_BI(self, test_tensor: torch.Tensor, keepdim, correction):
        self._test_var_tosa_BI_pipeline(self.Var(keepdim, correction), (test_tensor,))

    @parameterized.expand(Var.test_parameters)
    def test_var_u55_BI(self, test_tensor: torch.Tensor, keepdim, correction):
        self._test_var_ethosu_BI_pipeline(
            self.Var(keepdim, correction),
            common.get_u55_compile_spec(),
            (test_tensor,),
        )

    @parameterized.expand(Var.test_parameters)
    def test_var_u85_BI(self, test_tensor: torch.Tensor, keepdim, correction):
        self._test_var_ethosu_BI_pipeline(
            self.Var(keepdim, correction),
            common.get_u85_compile_spec(),
            (test_tensor,),
        )

    @parameterized.expand(VarDim.test_parameters)
    def test_var_dim_tosa_MI(self, test_tensor: torch.Tensor, dim, keepdim, unbiased):
        self._test_var_tosa_MI_pipeline(
            self.VarDim(dim, keepdim, unbiased), (test_tensor,)
        )

    @parameterized.expand(VarDim.test_parameters)
    def test_var_dim_tosa_BI(self, test_tensor: torch.Tensor, dim, keepdim, unbiased):
        self._test_var_tosa_BI_pipeline(
            self.VarDim(dim, keepdim, unbiased), (test_tensor,)
        )

    @parameterized.expand(VarDim.test_parameters_u55)
    def test_var_dim_u55_BI(self, test_tensor: torch.Tensor, dim, keepdim, unbiased):
        self._test_var_ethosu_BI_pipeline(
            self.VarDim(dim, keepdim, unbiased),
            common.get_u55_compile_spec(),
            (test_tensor,),
        )

    @parameterized.expand(VarDim.test_parameters)
    def test_var_dim_u85_BI(self, test_tensor: torch.Tensor, dim, keepdim, unbiased):
        self._test_var_ethosu_BI_pipeline(
            self.VarDim(dim, keepdim, unbiased),
            common.get_u85_compile_spec(),
            (test_tensor,),
        )

    @parameterized.expand(VarCorrection.test_parameters)
    def test_var_correction_tosa_MI(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_tosa_MI_pipeline(
            self.VarCorrection(dim, keepdim, correction), (test_tensor,)
        )

    @parameterized.expand(VarCorrection.test_parameters)
    def test_var_correction_tosa_BI(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_tosa_BI_pipeline(
            self.VarCorrection(dim, keepdim, correction), (test_tensor,)
        )

    @parameterized.expand(VarCorrection.test_parameters)
    def test_var_correction_u55_BI(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_ethosu_BI_pipeline(
            self.VarCorrection(dim, keepdim, correction),
            common.get_u55_compile_spec(),
            (test_tensor,),
        )

    @parameterized.expand(VarCorrection.test_parameters)
    def test_var_correction_u85_BI(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_ethosu_BI_pipeline(
            self.VarCorrection(dim, keepdim, correction),
            common.get_u85_compile_spec(),
            (test_tensor,),
        )
