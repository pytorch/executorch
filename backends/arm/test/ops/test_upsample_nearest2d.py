# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Optional, Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized


test_data_suite = [
    # (test_name, test_data, size, scale_factor, compare_outputs)
    ("rand_double_scale", torch.rand(2, 4, 8, 3), None, 2.0, True),
    ("rand_double_scale_one_dim", torch.rand(2, 4, 8, 3), None, (1.0, 2.0), True),
    ("rand_double_size", torch.rand(2, 4, 8, 3), (16, 6), None, True),
    ("rand_one_double_scale", torch.rand(2, 4, 1, 1), None, 2.0, True),
    ("rand_one_double_size", torch.rand(2, 4, 1, 1), (2, 2), None, True),
    ("rand_one_same_scale", torch.rand(2, 4, 1, 1), None, 1.0, True),
    ("rand_one_same_size", torch.rand(2, 4, 1, 1), (1, 1), None, True),
    # Can't compare outputs as the rounding when selecting the nearest pixel is
    # different between PyTorch and TOSA. Just check the legalization went well.
    # TODO Improve the test infrastructure to support more in depth verification
    # of the TOSA legalization results.
    ("rand_half_scale", torch.rand(2, 4, 8, 6), None, 0.5, False),
    ("rand_half_size", torch.rand(2, 4, 8, 6), (4, 3), None, False),
    ("rand_one_and_half_scale", torch.rand(2, 4, 8, 3), None, 1.5, False),
    ("rand_one_and_half_size", torch.rand(2, 4, 8, 3), (12, 4), None, False),
]


class TestUpsampleNearest2d(unittest.TestCase):
    class UpsamplingNearest2d(torch.nn.Module):
        def __init__(
            self,
            size: Optional[Tuple[int]],
            scale_factor: Optional[float | Tuple[float]],
        ):
            super().__init__()
            self.upsample = torch.nn.UpsamplingNearest2d(  # noqa: TOR101
                size=size, scale_factor=scale_factor
            )

        def forward(self, x):
            return self.upsample(x)

    class Upsample(torch.nn.Module):
        def __init__(
            self,
            size: Optional[Tuple[int]],
            scale_factor: Optional[float | Tuple[float]],
        ):
            super().__init__()
            self.upsample = torch.nn.Upsample(
                size=size, scale_factor=scale_factor, mode="nearest"
            )

        def forward(self, x):
            return self.upsample(x)

    class Interpolate(torch.nn.Module):
        def __init__(
            self,
            size: Optional[Tuple[int]],
            scale_factor: Optional[float | Tuple[float]],
        ):
            super().__init__()
            self.upsample = lambda x: torch.nn.functional.interpolate(
                x, size=size, scale_factor=scale_factor, mode="nearest"
            )

        def forward(self, x):
            return self.upsample(x)

    def _test_upsample_nearest_2d_tosa_MI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.tensor],
        compare_outputs: bool,
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .check(["torch.ops.aten.upsample_nearest2d.vec"])
            .check_not(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.aten.upsample_nearest2d.vec"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

        if compare_outputs:
            tester.run_method_and_compare_outputs(inputs=test_data)

    def _test_upsample_nearest_2d_tosa_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.tensor],
        compare_outputs: bool,
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
            )
            .quantize()
            .export()
            .check(["torch.ops.aten.upsample_nearest2d.vec"])
            .check(["torch.ops.quantized_decomposed"])
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.aten.upsample_nearest2d.vec"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

        if compare_outputs:
            tester.run_method_and_compare_outputs(inputs=test_data)

    @parameterized.expand(test_data_suite)
    def test_upsample_nearest_2d_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        size: Optional[Tuple[int]],
        scale_factor: Optional[float | Tuple[float]],
        compare_outputs: bool,
    ):
        self._test_upsample_nearest_2d_tosa_MI_pipeline(
            self.UpsamplingNearest2d(size, scale_factor), (test_data,), compare_outputs
        )
        self._test_upsample_nearest_2d_tosa_MI_pipeline(
            self.Upsample(size, scale_factor), (test_data,), compare_outputs
        )
        self._test_upsample_nearest_2d_tosa_MI_pipeline(
            self.Interpolate(size, scale_factor), (test_data,), compare_outputs
        )

    @parameterized.expand(test_data_suite)
    def test_upsample_nearest_2d_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        size: Optional[Tuple[int]],
        scale_factor: Optional[float | Tuple[float]],
        compare_outputs: bool,
    ):
        self._test_upsample_nearest_2d_tosa_BI_pipeline(
            self.UpsamplingNearest2d(size, scale_factor), (test_data,), compare_outputs
        )
        self._test_upsample_nearest_2d_tosa_BI_pipeline(
            self.Upsample(size, scale_factor), (test_data,), compare_outputs
        )
        self._test_upsample_nearest_2d_tosa_BI_pipeline(
            self.Interpolate(size, scale_factor), (test_data,), compare_outputs
        )
