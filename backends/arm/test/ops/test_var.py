# Copyright 2024 Arm Limited and/or its affiliates.
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
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
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

        def forward(
            self,
            x: torch.Tensor,
            keepdim: bool = True,
            correction: int = 0,
        ):
            return x.var(keepdim=keepdim, correction=correction)

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

        def forward(
            self,
            x: torch.Tensor,
            dim: int = -1,
            keepdim: bool = True,
            unbiased: bool = False,
        ):
            return x.var(dim=dim, keepdim=keepdim, unbiased=unbiased)

    class VarCorrection(torch.nn.Module):
        test_parameters = [
            (torch.randn(1, 50, 10, 20), (-1, -2), True, 0),
            (torch.rand(1, 50, 10), (-2), True, 0),
            (torch.randn(1, 30, 15, 20), (-1, -2, -3), True, 1),
            (torch.rand(1, 50, 10, 20), (-1, -2), True, 0.5),
        ]

        def forward(
            self,
            x: torch.Tensor,
            dim: int | tuple[int] = -1,
            keepdim: bool = True,
            correction: int = 0,
        ):
            return x.var(dim=dim, keepdim=keepdim, correction=correction)

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
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
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
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
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
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
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
        )

    @parameterized.expand(Var.test_parameters)
    def test_var_tosa_MI(self, test_tensor: torch.Tensor, keepdim, correction):
        self._test_var_tosa_MI_pipeline(self.Var(), (test_tensor, keepdim, correction))

    @parameterized.expand(Var.test_parameters)
    def test_var_tosa_BI(self, test_tensor: torch.Tensor, keepdim, correction):
        self._test_var_tosa_BI_pipeline(self.Var(), (test_tensor, keepdim, correction))

    # Expected to fail as this is not supported on u55.
    @parameterized.expand(Var.test_parameters)
    @unittest.expectedFailure
    def test_var_u55_BI_xfails(self, test_tensor: torch.Tensor, keepdim, correction):
        self._test_var_ethosu_BI_pipeline(
            self.Var(),
            common.get_u55_compile_spec(),
            (test_tensor, keepdim, correction),
        )

    @parameterized.expand(Var.test_parameters)
    def test_var_u85_BI(self, test_tensor: torch.Tensor, keepdim, correction):
        self._test_var_ethosu_BI_pipeline(
            self.Var(),
            common.get_u85_compile_spec(),
            (test_tensor, keepdim, correction),
        )

    @parameterized.expand(VarDim.test_parameters)
    def test_var_dim_tosa_MI(self, test_tensor: torch.Tensor, dim, keepdim, correction):
        self._test_var_tosa_MI_pipeline(
            self.VarDim(), (test_tensor, dim, keepdim, correction)
        )

    @parameterized.expand(VarDim.test_parameters)
    def test_var_dim_tosa_BI(self, test_tensor: torch.Tensor, dim, keepdim, correction):
        self._test_var_tosa_BI_pipeline(
            self.VarDim(), (test_tensor, dim, keepdim, correction)
        )

    @parameterized.expand(VarDim.test_parameters_u55)
    def test_var_dim_u55_BI(self, test_tensor: torch.Tensor, dim, keepdim, correction):
        self._test_var_ethosu_BI_pipeline(
            self.VarDim(),
            common.get_u55_compile_spec(),
            (test_tensor, dim, keepdim, correction),
        )

    # Expected to fail as this is not supported on u55.
    @parameterized.expand(VarDim.test_parameters_u55_xfails)
    @unittest.expectedFailure
    def test_var_dim_u55_BI_xfails(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_ethosu_BI_pipeline(
            self.VarDim(),
            common.get_u55_compile_spec(),
            (test_tensor, dim, keepdim, correction),
        )

    @parameterized.expand(VarDim.test_parameters)
    def test_var_dim_u85_BI(self, test_tensor: torch.Tensor, dim, keepdim, correction):
        self._test_var_ethosu_BI_pipeline(
            self.VarDim(),
            common.get_u85_compile_spec(),
            (test_tensor, dim, keepdim, correction),
        )

    @parameterized.expand(VarCorrection.test_parameters)
    def test_var_correction_tosa_MI(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_tosa_MI_pipeline(
            self.VarCorrection(), (test_tensor, dim, keepdim, correction)
        )

    @parameterized.expand(VarCorrection.test_parameters)
    def test_var_correction_tosa_BI(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_tosa_BI_pipeline(
            self.VarCorrection(), (test_tensor, dim, keepdim, correction)
        )

    # Expected to fail as this is not supported on u55.
    @parameterized.expand(VarCorrection.test_parameters)
    @unittest.expectedFailure
    def test_var_correction_u55_BI_xfails(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_ethosu_BI_pipeline(
            self.VarCorrection(),
            common.get_u55_compile_spec(),
            (test_tensor, dim, keepdim, correction),
        )

    @parameterized.expand(VarCorrection.test_parameters)
    def test_var_correction_u85_BI(
        self, test_tensor: torch.Tensor, dim, keepdim, correction
    ):
        self._test_var_ethosu_BI_pipeline(
            self.VarCorrection(),
            common.get_u85_compile_spec(),
            (test_tensor, dim, keepdim, correction),
        )
