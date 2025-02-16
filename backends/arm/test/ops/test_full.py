# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the full op which creates a tensor of a given shape filled with a given value.
# The shape and value are set at compile time, i.e. can't be set by a tensor input.
#

import unittest

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


class TestFull(unittest.TestCase):
    """Tests the full op which creates a tensor of a given shape filled with a given value."""

    class Full(torch.nn.Module):
        # A single full op
        def forward(self):
            return torch.full((3, 3), 4.5)

    class AddConstFull(torch.nn.Module):
        # Input + a full with constant value.
        def forward(self, x: torch.Tensor):
            return torch.full((2, 2, 3, 3), 4.5, dtype=torch.float32) + x

    class AddVariableFull(torch.nn.Module):
        sizes: list[tuple[int, ...]] = [
            (5,),
            (5, 5),
            (5, 5, 5),
            (1, 5, 5, 5),
        ]
        test_parameters = [((torch.randn(n) * 10 - 5, 3.2),) for n in sizes]

        def forward(self, x: torch.Tensor, y):
            # Input + a full with the shape from the input and a given value 'y'.
            return x + torch.full(x.shape, y)

    class FullLike(torch.nn.Module):
        """Since full_like is replaced with full, we only need to test on reference model, not FVP."""

        test_parameters = [
            ((torch.randn(2, 2, 2, 2) * 50, 3.2),),
            ((torch.randn(2, 2, 2, 2) * 50, 3),),
            (((torch.randn(2, 2, 2, 2) * 50).to(torch.int32), 3.2),),
            (((torch.randn(2, 2, 2, 2) * 50).to(torch.int32), 3),),
        ]

        def forward(self, input_tensor: torch.Tensor, value):
            # Our backend can't handle tensors without users, which input_tensor doesn't have
            # when the full_like is converted to a full. Therefore involve it in the output.
            return input_tensor + torch.full_like(input_tensor, value)

    def _test_full_tosa_MI_pipeline(
        self,
        module: torch.nn.Module,
        example_data: Tuple,
        test_data: Tuple | None = None,
    ):
        if test_data is None:
            test_data = example_data
        (
            ArmTester(
                module,
                example_inputs=example_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_full_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_full_tosa_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple,
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_full_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_full_tosa_ethos_pipeline(
        self, compile_spec: list[CompileSpec], module: torch.nn.Module, test_data: Tuple
    ):
        tester = (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check_not(["executorch_exir_dialects_edge__ops_aten_full_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    def _test_full_tosa_u55_pipeline(self, module: torch.nn.Module, test_data: Tuple):
        self._test_full_tosa_ethos_pipeline(
            common.get_u55_compile_spec(), module, test_data
        )

    def _test_full_tosa_u85_pipeline(self, module: torch.nn.Module, test_data: Tuple):
        self._test_full_tosa_ethos_pipeline(
            common.get_u85_compile_spec(), module, test_data
        )

    def test_only_full_tosa_MI(self):
        self._test_full_tosa_MI_pipeline(self.Full(), ())

    def test_const_full_tosa_MI(self):
        _input = torch.rand((2, 2, 3, 3)) * 10
        self._test_full_tosa_MI_pipeline(self.AddConstFull(), (_input,))

    @parameterized.expand(FullLike.test_parameters)
    def test_full_like_tosa_MI(self, test_tensor: Tuple):
        self._test_full_tosa_MI_pipeline(self.FullLike(), test_tensor)

    def test_const_full_nhwc_tosa_BI(self):
        _input = torch.rand((2, 2, 3, 3)) * 10
        self._test_full_tosa_BI_pipeline(self.AddConstFull(), (_input,))

    @parameterized.expand(AddVariableFull.test_parameters)
    def test_full_tosa_MI(self, test_tensor: Tuple):
        self._test_full_tosa_MI_pipeline(
            self.AddVariableFull(), example_data=test_tensor
        )

    @parameterized.expand(AddVariableFull.test_parameters)
    def test_full_tosa_BI(self, test_tensor: Tuple):
        self._test_full_tosa_BI_pipeline(self.AddVariableFull(), test_tensor)

    @parameterized.expand(FullLike.test_parameters)
    def test_full_like_tosa_BI(self, test_tensor: Tuple):
        self._test_full_tosa_BI_pipeline(self.FullLike(), test_tensor)

    @parameterized.expand(AddVariableFull.test_parameters)
    @pytest.mark.corstone_fvp
    def test_full_u55_BI(self, test_tensor: Tuple):
        self._test_full_tosa_u55_pipeline(
            self.AddVariableFull(),
            test_tensor,
        )

    @parameterized.expand(AddVariableFull.test_parameters)
    @pytest.mark.corstone_fvp
    def test_full_u85_BI(self, test_tensor: Tuple):
        self._test_full_tosa_u85_pipeline(
            self.AddVariableFull(),
            test_tensor,
        )

    # This fails since full outputs int64 by default if 'fill_value' is integer, which our backend doesn't support.
    @unittest.expectedFailure
    def test_integer_value(self):
        _input = torch.ones((2, 2))
        integer_fill_value = 1
        self._test_full_tosa_MI_pipeline(
            self.AddVariableFull(), example_data=(_input, integer_fill_value)
        )

    # This fails since the fill value in the full tensor is set at compile time by the example data (1.).
    # Test data tries to set it again at runtime (to 2.) but it doesn't do anything.
    # In eager mode, the fill value can be set at runtime, causing the outputs to not match.
    @unittest.expectedFailure
    def test_set_value_at_runtime(self):
        _input = torch.ones((2, 2))
        example_fill_value = 1.0
        test_fill_value = 2.0
        self._test_full_tosa_MI_pipeline(
            self.AddVariableFull(),
            example_data=(_input, example_fill_value),
            test_data=(_input, test_fill_value),
        )
