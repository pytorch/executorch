# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import List, Tuple, Union

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized


test_data_suite = [
    # (test_name, test_data, [normalized_shape, eps, elementwise_affine, has_bias] )
    ("randn_last_dim", torch.randn(1, 5, 5, 5), [[5]]),
    ("rand_last_two_dims", torch.rand(1, 5, 5, 5), [[5, 5]]),
    (
        "rand_last_two_dims_not_elementwise_affine",
        torch.rand(1, 5, 5, 5),
        [[5, 5], 1e-5, False],
    ),
    (
        "rand_last_two_dims_not_elementwise_affine_no_bias",
        torch.rand(1, 5, 5, 5),
        [[5, 5], 1e-5, False, False],
    ),
    ("randn_last_three_dims", torch.randn(1, 15, 10, 5), [[15, 10, 5]]),
    (
        "randn_last_three_dims_no_bias",
        torch.randn(1, 15, 10, 5),
        [[15, 10, 5], 1e-2, False, False],
    ),
]


class TestLayerNorm(unittest.TestCase):

    class LayerNorm(torch.nn.Module):

        def __init__(
            self,
            normalized_shape: Union[int, List[int]],
            eps: float = 1e-5,
            elementwise_affine: bool = True,
            has_bias: bool = True,
        ):
            super().__init__()
            self.layer_norm = torch.nn.LayerNorm(
                normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                bias=has_bias,
            )
            if elementwise_affine:
                self.layer_norm.weight = torch.nn.Parameter(
                    torch.ones(normalized_shape)
                )
                if has_bias:
                    self.layer_norm.bias = torch.nn.Parameter(
                        torch.rand(normalized_shape)
                    )

        def forward(self, x):
            return self.layer_norm(x)

    def _test_layernorm_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                model=module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+MI", permute_memory_to_nhwc=True
                ),
            )
            .export()
            .check(["torch.ops.aten.layer_norm.default"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["torch.ops.aten.layer_norm.default"])
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_layernorm_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                model=module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+BI", permute_memory_to_nhwc=True
                ),
            )
            .quantize()
            .check_not(["torch.ops.aten.layer_norm.default"])
            .export()
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_layernorm_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.Tensor],
    ):
        tester = (
            ArmTester(
                model=module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .check_not(["torch.ops.aten.layer_norm.default"])
            .export()
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_layer_norm_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params,
    ):
        self._test_layernorm_tosa_MI_pipeline(
            self.LayerNorm(*model_params), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    def test_layer_norm_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params,
    ):
        self._test_layernorm_tosa_BI_pipeline(
            self.LayerNorm(*model_params), (test_data,)
        )

    # Numerical issues on FVP likely due to mul op, MLETORCH-521
    # Skip tests that require transposes.
    @parameterized.expand(test_data_suite)
    @pytest.mark.corstone_fvp
    @unittest.expectedFailure
    def test_layer_norm_u55_BI_xfails(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params,
    ):
        self._test_layernorm_ethosu_BI_pipeline(
            self.LayerNorm(*model_params), common.get_u55_compile_spec(), (test_data,)
        )

    # Numerical issues on FVP likely due to mul op, MLETORCH-521
    @parameterized.expand(test_data_suite[:-2])
    @pytest.mark.corstone_fvp
    @unittest.expectedFailure
    def test_layer_norm_u85_BI_xfails(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params,
    ):
        self._test_layernorm_ethosu_BI_pipeline(
            self.LayerNorm(*model_params), common.get_u85_compile_spec(), (test_data,)
        )

    @parameterized.expand(test_data_suite[-2:])
    @pytest.mark.corstone_fvp
    def test_layer_norm_u85_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        model_params,
    ):
        self._test_layernorm_ethosu_BI_pipeline(
            self.LayerNorm(*model_params), common.get_u85_compile_spec(), (test_data,)
        )
