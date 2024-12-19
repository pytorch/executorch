# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.ops.test_conv1d import Conv1d
from executorch.backends.arm.test.ops.test_conv2d import Conv2d

from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized


"""
The configuration when
  groups == in_channels and
  out_channels = K * in_channels
  where K is a positive integer
is termed in literature as depthwise convolution.
"""

dw_conv1d_3_1x3x14_gp3_st1 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=7,
    stride=1,
    groups=3,
    length=14,
    batches=1,
    padding=3,
)

dw_conv1d_2_1x6x4_gp6_st1 = Conv1d(
    in_channels=6,
    out_channels=12,
    kernel_size=2,
    stride=1,
    groups=6,
    padding=0,
    length=4,
    batches=1,
)

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

dw_conv1d_3_1x3x256_gp3_st1 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=3,
    stride=1,
    groups=3,
    padding=0,
    length=256,
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

two_dw_conv1d = Conv1d(
    nbr_conv=2,
    length=64,
    in_channels=[4, 8],
    out_channels=[8, 24],
    kernel_size=[3, 3],
    stride=[1, 1],
    padding=[0, 0],
    groups=[4, 8],
    bias=[True, True],
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
testsuite_conv2d = [
    ("2x2_1x6x4x4_gp6_st1", dw_conv2d_2x2_1x6x4x4_gp6_st1),
    ("3x3_1x3x256x256_gp3_st1", dw_conv2d_3x3_1x3x256x256_gp3_st1),
    ("3x3_1x4x256x256_gp4_st1", dw_conv2d_3x3_1x4x256x256_gp4_st1),
    ("3x3_2x8x198x198_gp8_st3", dw_conv2d_3x3_2x8x198x198_gp8_st3),
    ("3x3_1x4x256x256_gp4_nobias", dw_conv2d_3x3_1x4x256x256_gp4_nobias),
    ("two_dw_conv2d", two_dw_conv2d),
]

testsuite_conv2d_u85 = [
    ("2x2_1x6x4x4_gp6_st1", dw_conv2d_2x2_1x6x4x4_gp6_st1),
    ("3x3_1x3x256x256_gp3_st1", dw_conv2d_3x3_1x3x256x256_gp3_st1),
    ("3x3_1x4x256x256_gp4_st1", dw_conv2d_3x3_1x4x256x256_gp4_st1),
    ("3x3_1x4x256x256_gp4_nobias", dw_conv2d_3x3_1x4x256x256_gp4_nobias),
]

testsuite_conv2d_u85_xfails = [
    ("3x3_2x8x198x198_gp8_st3", dw_conv2d_3x3_2x8x198x198_gp8_st3),
    ("two_dw_conv2d", two_dw_conv2d),
]


testsuite_conv1d = [
    ("2_1x6x4_gp6_st1", dw_conv1d_2_1x6x4_gp6_st1),
    ("two_dw_conv1d", two_dw_conv1d),
    ("3_1x3x256_gp3_st1", dw_conv1d_3_1x3x256_gp3_st1),
    ("3_1x3x14_gp3_st1", dw_conv1d_3_1x3x14_gp3_st1),
]


class TestDepthwiseConv(unittest.TestCase):
    """Tests Conv1D and Conv2D where groups == in_channels and out_channels = K * in_channels. This
    is a special case enables depthwise convolution."""

    def _test_dw_conv_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+MI", permute_memory_to_nhwc=True
                ),
            )
            .export()
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_dw_conv_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80+BI", permute_memory_to_nhwc=True
                ),
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

    def _test_dw_conv_ethos_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.Tensor],
    ):
        tester = (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=compile_spec,
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(testsuite_conv1d + testsuite_conv2d)
    def test_dw_conv_tosa_MI(self, test_name: str, model: torch.nn.Module):
        self._test_dw_conv_tosa_MI_pipeline(model, model.get_inputs())

    # TODO: Investigate flakyness (MLTORCH-307)
    @parameterized.expand(testsuite_conv1d + testsuite_conv2d)
    def test_dw_conv_tosa_BI(self, test_name: str, model: torch.nn.Module):
        self._test_dw_conv_tosa_BI_pipeline(model, model.get_inputs())

    testsuite_conv2d.remove(
        ("3x3_1x3x256x256_gp3_st1", dw_conv2d_3x3_1x3x256x256_gp3_st1)
    )  # Works

    @parameterized.expand(testsuite_conv2d, skip_on_empty=True)
    @pytest.mark.corstone_fvp
    @unittest.expectedFailure
    def test_dw_conv2d_u55_BI(
        self, test_name: str, model: torch.nn.Module, set_quantize_io: bool = False
    ):
        self._test_dw_conv_ethos_BI_pipeline(
            model,
            common.get_u55_compile_spec(
                permute_memory_to_nhwc=True, quantize_io=set_quantize_io
            ),
            model.get_inputs(),
        )

    # Expected to fail as conv1d needs transpose which is not supported
    # on u55.
    @parameterized.expand(testsuite_conv1d, skip_on_empty=True)
    @pytest.mark.corstone_fvp
    @unittest.expectedFailure
    def test_dw_conv1d_u55_BI(
        self, test_name: str, model: torch.nn.Module, set_quantize_io: bool = False
    ):
        self._test_dw_conv_ethos_BI_pipeline(
            model,
            common.get_u55_compile_spec(
                permute_memory_to_nhwc=True, quantize_io=set_quantize_io
            ),
            model.get_inputs(),
        )

    @parameterized.expand(testsuite_conv1d + testsuite_conv2d_u85)
    @pytest.mark.corstone_fvp
    def test_dw_conv_u85_BI(
        self, test_name: str, model: torch.nn.Module, set_quantize_io: bool = False
    ):
        self._test_dw_conv_ethos_BI_pipeline(
            model,
            common.get_u85_compile_spec(
                permute_memory_to_nhwc=True, quantize_io=set_quantize_io
            ),
            model.get_inputs(),
        )

    # All test cases except 3x3_1x3x256x256_gp3_st1 have numerical issues on FVP. MLETORCH-520
    @parameterized.expand(testsuite_conv2d_u85_xfails)
    @pytest.mark.corstone_fvp
    @conftest.expectedFailureOnFVP
    def test_dw_conv_u85_BI_xfails(
        self, test_name: str, model: torch.nn.Module, set_quantize_io: bool = False
    ):
        self._test_dw_conv_ethos_BI_pipeline(
            model,
            common.get_u85_compile_spec(
                permute_memory_to_nhwc=True, quantize_io=set_quantize_io
            ),
            model.get_inputs(),
        )
