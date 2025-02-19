# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Test the pad_constant_nd op which pads the input tensor at specific dimension(s).
#
import unittest
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

test_data_suite = [
    ("4dim_last1dim", torch.rand(1, 1, 16, 16), (1, 1, 0, 0, 0, 0, 0, 0), 1),
    ("4dim_last2dim", torch.rand(1, 1, 16, 16), (1, 0, 1, 0, 0, 0, 0, 0), 2),
    ("4dim_last3dim", torch.rand(1, 1, 16, 16), (1, 1, 0, 2, 0, 2, 0, 0), 3),
    ("4dim_last4dim", torch.rand(1, 1, 16, 16), (1, 0, 1, 1, 0, 2, 0, 2), 4),
    ("3dim_last1dim", torch.rand(1, 1, 16), (1, 1, 0, 0, 0, 0), 1),
    ("3dim_last2dim", torch.rand(1, 1, 16), (1, 0, 1, 1, 0, 0), 2),
    ("3dim_last3dim", torch.rand(1, 1, 16), (1, 0, 1, 0, 1, 1), 3),
    ("2dim_last1dim", torch.rand(1, 1, 16), (1, 1, 0, 0), 1),
    ("2dim_last2dim", torch.rand(1, 1, 16), (1, 0, 1, 1), 2),
]


class TestConstantPadND(unittest.TestCase):
    """Tests pad."""

    class ConstantPadND(torch.nn.Module):
        def __init__(self, pad: Tuple, value: float | None = None):
            super().__init__()
            self.dim = len(pad) // 2
            self.value = value
            in_channels = 1
            # Only apply conv2d when the input dim = 4.
            if self.dim == 4:
                in_channels += pad[-3] + pad[-4]

                self.conv2d = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=3,
                    kernel_size=3,
                    bias=True,
                    stride=(2, 2),
                    padding=0,
                )

                in_channels = 3
                in_channels += pad[-3] + pad[-4]
                self.conv2d_1 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=3,
                    kernel_size=3,
                    bias=True,
                    padding="same",
                )

            nonzero_idx = len(pad)
            for i in range(0, len(pad), 2):
                if pad[i] + pad[i + 1] == 0:
                    nonzero_idx = i
                    break
            self.pad = pad[:nonzero_idx]
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor):
            x = F.pad(x, pad=self.pad, mode="constant", value=self.value)
            if self.dim == 4:
                x = self.conv2d(x)
            x = self.relu(x)

            x = F.pad(x, pad=self.pad, mode="constant", value=self.value)
            if self.dim == 4:
                x = self.conv2d_1(x)
            x = self.sigmoid(x)
            return x

    def _test_constant_pad_nd_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+MI"),
            )
            .export()
            .check_count({"torch.ops.aten.pad.default": 2})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_constant_pad_nd_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80+BI"),
            )
            .quantize()
            .export()
            .check_count({"torch.ops.aten.pad.default": 2})
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    @parameterized.expand(test_data_suite)
    def test_constant_pad_nd_tosa_MI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        padding: Tuple,
        value: float | None = None,
    ):
        self._test_constant_pad_nd_tosa_MI_pipeline(
            self.ConstantPadND(padding, value), (test_data,)
        )

    @parameterized.expand(test_data_suite)
    def test_constant_pad_nd_tosa_BI(
        self,
        test_name: str,
        test_data: torch.Tensor,
        padding: Tuple,
        value: float | None = None,
    ):
        self._test_constant_pad_nd_tosa_BI_pipeline(
            self.ConstantPadND(padding, value), (test_data,)
        )
