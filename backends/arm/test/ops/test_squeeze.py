# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the squeeze op which squeezes a given dimension with size 1 into a lower ranked tensor.
#

import unittest
from typing import Optional, Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from executorch.exir.backend.compile_spec_schema import CompileSpec
from parameterized import parameterized


class TestSqueeze(unittest.TestCase):
    class SqueezeDim(torch.nn.Module):
        test_parameters: list[tuple[torch.Tensor, int]] = [
            (torch.randn(1, 1, 5), -2),
            (torch.randn(1, 2, 3, 1), 3),
            (torch.randn(1, 5, 1, 5), -2),
        ]

        def forward(self, x: torch.Tensor, dim: int):
            return x.squeeze(dim)

    class SqueezeDims(torch.nn.Module):
        test_parameters: list[tuple[torch.Tensor, tuple[int]]] = [
            (torch.randn(1, 1, 5), (0, 1)),
            (torch.randn(1, 5, 5, 1), (0, -1)),
            (torch.randn(1, 5, 1, 5), (0, -2)),
        ]

        def forward(self, x: torch.Tensor, dims: tuple[int]):
            return x.squeeze(dims)

    class Squeeze(torch.nn.Module):
        test_parameters: list[tuple[torch.Tensor]] = [
            (torch.randn(1, 1, 5),),
            (torch.randn(1, 5, 5, 1),),
            (torch.randn(1, 5, 1, 5),),
        ]

        def forward(self, x: torch.Tensor):
            return x.squeeze()

    def _test_squeeze_tosa_MI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor, Optional[tuple[int]]],
        export_target: str,
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check_count({export_target: 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_squeeze_tosa_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor, Optional[tuple[int]]],
        export_target: str,
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check_count({export_target: 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_squeeze_ethosu_BI_pipeline(
        self,
        compile_spec: CompileSpec,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor, Optional[tuple[int]]],
        export_target: str,
    ):
        (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize()
            .export()
            .check_count({export_target: 1})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Squeeze.test_parameters)
    def test_squeeze_tosa_MI(
        self,
        test_tensor: torch.Tensor,
    ):
        self._test_squeeze_tosa_MI_pipeline(
            self.Squeeze(), (test_tensor,), "torch.ops.aten.squeeze.default"
        )

    @parameterized.expand(Squeeze.test_parameters)
    def test_squeeze_tosa_BI(
        self,
        test_tensor: torch.Tensor,
    ):
        self._test_squeeze_tosa_BI_pipeline(
            self.Squeeze(), (test_tensor,), "torch.ops.aten.squeeze.default"
        )

    @parameterized.expand(Squeeze.test_parameters)
    def test_squeeze_u55_BI(
        self,
        test_tensor: torch.Tensor,
    ):
        self._test_squeeze_ethosu_BI_pipeline(
            common.get_u55_compile_spec(permute_memory_to_nhwc=False),
            self.Squeeze(),
            (test_tensor,),
            "torch.ops.aten.squeeze.default",
        )

    @parameterized.expand(Squeeze.test_parameters)
    def test_squeeze_u85_BI(
        self,
        test_tensor: torch.Tensor,
    ):
        self._test_squeeze_ethosu_BI_pipeline(
            common.get_u85_compile_spec(permute_memory_to_nhwc=True),
            self.Squeeze(),
            (test_tensor,),
            "torch.ops.aten.squeeze.default",
        )

    @parameterized.expand(SqueezeDim.test_parameters)
    def test_squeeze_dim_tosa_MI(self, test_tensor: torch.Tensor, dim: int):
        self._test_squeeze_tosa_MI_pipeline(
            self.SqueezeDim(), (test_tensor, dim), "torch.ops.aten.squeeze.dim"
        )

    @parameterized.expand(SqueezeDim.test_parameters)
    def test_squeeze_dim_tosa_BI(self, test_tensor: torch.Tensor, dim: int):
        self._test_squeeze_tosa_BI_pipeline(
            self.SqueezeDim(), (test_tensor, dim), "torch.ops.aten.squeeze.dim"
        )

    @parameterized.expand(SqueezeDim.test_parameters)
    def test_squeeze_dim_u55_BI(self, test_tensor: torch.Tensor, dim: int):
        self._test_squeeze_ethosu_BI_pipeline(
            common.get_u55_compile_spec(permute_memory_to_nhwc=False),
            self.SqueezeDim(),
            (test_tensor, dim),
            "torch.ops.aten.squeeze.dim",
        )

    @parameterized.expand(SqueezeDim.test_parameters)
    def test_squeeze_dim_u85_BI(self, test_tensor: torch.Tensor, dim: int):
        self._test_squeeze_ethosu_BI_pipeline(
            common.get_u85_compile_spec(permute_memory_to_nhwc=True),
            self.SqueezeDim(),
            (test_tensor, dim),
            "torch.ops.aten.squeeze.dim",
        )

    @parameterized.expand(SqueezeDims.test_parameters)
    def test_squeeze_dims_tosa_MI(self, test_tensor: torch.Tensor, dims: tuple[int]):
        self._test_squeeze_tosa_MI_pipeline(
            self.SqueezeDims(), (test_tensor, dims), "torch.ops.aten.squeeze.dims"
        )

    @parameterized.expand(SqueezeDims.test_parameters)
    def test_squeeze_dims_tosa_BI(self, test_tensor: torch.Tensor, dims: tuple[int]):
        self._test_squeeze_tosa_BI_pipeline(
            self.SqueezeDims(), (test_tensor, dims), "torch.ops.aten.squeeze.dims"
        )

    @parameterized.expand(SqueezeDims.test_parameters)
    def test_squeeze_dims_u55_BI(self, test_tensor: torch.Tensor, dims: tuple[int]):
        self._test_squeeze_ethosu_BI_pipeline(
            common.get_u55_compile_spec(permute_memory_to_nhwc=False),
            self.SqueezeDims(),
            (test_tensor, dims),
            "torch.ops.aten.squeeze.dims",
        )

    @parameterized.expand(SqueezeDims.test_parameters)
    def test_squeeze_dims_u85_BI(self, test_tensor: torch.Tensor, dims: tuple[int]):
        self._test_squeeze_ethosu_BI_pipeline(
            common.get_u85_compile_spec(),
            self.SqueezeDims(),
            (test_tensor, dims),
            "torch.ops.aten.squeeze.dims",
        )
