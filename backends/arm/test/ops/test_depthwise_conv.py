# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.ops.test_conv import Conv2d

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from parameterized import parameterized

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
The configuration when
  groups == in_channels and
  out_channels = K * in_channels
  where K is a positive integer
is termed in literature as depthwise convolution.
"""
dw_conv2d_2x2_1x6x4x4_gp6_st1 = Conv2d(
    in_channels=6,
    out_channels=12,
    kernel_size=(2, 2),
    stride=(1, 1),
    groups=6,
    padding=0,
    width=4,
    height=4,
    batches=1,
)

dw_conv2d_3x3_1x3x256x256_gp3_st1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3),
    stride=(1, 1),
    groups=3,
    padding=0,
    width=256,
    height=256,
    batches=1,
)

dw_conv2d_3x3_1x4x256x256_gp4_st1 = Conv2d(
    in_channels=4,
    out_channels=8,
    kernel_size=(3, 3),
    stride=(1, 1),
    groups=4,
    padding=0,
    width=256,
    height=256,
    batches=1,
)

dw_conv2d_3x3_2x8x198x198_gp8_st3 = Conv2d(
    in_channels=8,
    out_channels=16,
    kernel_size=(3, 3),
    stride=3,
    groups=8,
    padding=0,
    width=198,
    height=198,
    batches=2,
)

dw_conv2d_3x3_1x4x256x256_gp4_nobias = Conv2d(
    in_channels=4,
    out_channels=8,
    kernel_size=(3, 3),
    stride=1,
    groups=4,
    bias=False,
    width=256,
    height=256,
    batches=1,
)

two_dw_conv2d = Conv2d(
    nbr_conv=2,
    width=64,
    height=64,
    in_channels=[4, 8],
    out_channels=[8, 24],
    kernel_size=[(3, 3), (3, 3)],
    stride=[1, 1],
    padding=[0, 0],
    groups=[4, 8],
    bias=[True, True],
    batches=2,
)

# Shenanigan to get a nicer output when test fails.
testsuite = [
    ("2x2_1x6x4x4_gp6_st1", dw_conv2d_2x2_1x6x4x4_gp6_st1),
    ("3x3_1x3x256x256_gp3_st1", dw_conv2d_3x3_1x3x256x256_gp3_st1),
    ("3x3_1x4x256x256_gp4_st1", dw_conv2d_3x3_1x4x256x256_gp4_st1),
    ("3x3_2x8x198x198_gp8_st3", dw_conv2d_3x3_2x8x198x198_gp8_st3),
    ("3x3_1x4x256x256_gp4_nobias", dw_conv2d_3x3_1x4x256x256_gp4_nobias),
    ("two_dw_conv2d", two_dw_conv2d),
]

# Expected fails on Ethos-U55/U65. This is a known limitation.
# Check: https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/refs/heads/main/SUPPORTED_OPS.md
#   For depth multipliers > 1, IFM channels must be 1 and OFM channels must be
#   equal to the depth multiplier
# and
#   depthwise_multiplier = out_channels / in_channels
testsuite_u55 = testsuite.copy()
testsuite_u55.remove(("2x2_1x6x4x4_gp6_st1", dw_conv2d_2x2_1x6x4x4_gp6_st1))
testsuite_u55.remove(("3x3_1x4x256x256_gp4_st1", dw_conv2d_3x3_1x4x256x256_gp4_st1))
testsuite_u55.remove(("3x3_2x8x198x198_gp8_st3", dw_conv2d_3x3_2x8x198x198_gp8_st3))
testsuite_u55.remove(
    ("3x3_1x4x256x256_gp4_nobias", dw_conv2d_3x3_1x4x256x256_gp4_nobias)
)
testsuite_u55.remove(("two_dw_conv2d", two_dw_conv2d))

# Fails when enabling CompileSpec.set_quantize_io(True). MLETORCH-191.
testsuite_u55.remove(("3x3_1x3x256x256_gp3_st1", dw_conv2d_3x3_1x3x256x256_gp3_st1))

# Add failing test (set_quantize_io=True) temporarily to investigate
testsuite_u55.append(
    ("3x3_1x3x256x256_gp3_st1", dw_conv2d_3x3_1x3x256x256_gp3_st1, True)
)


class TestDepthwiseConv2D(unittest.TestCase):
    """Tests Conv2D where groups == in_channels and out_channels = K * in_channels. This
    is a special case enables depthwise convolution."""

    def _test_dw_conv2d_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(permute_memory_to_nhwc=True),
            )
            .export()
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_dw_conv2d_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(permute_memory_to_nhwc=True),
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_dw_conv2d_u55_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
        set_quantize_io: bool = False,
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(
                    permute_memory_to_nhwc=True, quantize_io=set_quantize_io
                ),
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(testsuite)
    def test_dw_conv2d_tosa_MI(self, test_name, model):
        self._test_dw_conv2d_tosa_MI_pipeline(model, model.get_inputs())

    # TODO: Investigate flakyness (MLTORCH-307)
    @parameterized.expand(testsuite)
    @pytest.mark.flaky(reruns=3)
    def test_dw_conv2d_tosa_BI(self, test_name, model):
        self._test_dw_conv2d_tosa_BI_pipeline(model, model.get_inputs())

    @parameterized.expand(testsuite_u55, skip_on_empty=True)
    @unittest.expectedFailure
    def test_dw_conv2d_u55_BI(self, test_name, model, set_quantize_io=False):
        self._test_dw_conv2d_u55_BI_pipeline(
            model, model.get_inputs(), set_quantize_io=set_quantize_io
        )
