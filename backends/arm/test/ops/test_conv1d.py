# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import List, Optional, Tuple, Union

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized


class Conv1d(torch.nn.Module):
    """
    Creates one or many chained 1D-convolutions. For multiple convolutions, the
    respective parameteres are provided as lists.
    """

    def __init__(
        self,
        inputs: Optional[torch.Tensor] = None,
        length=8,
        nbr_conv=1,  # Number of chained convs
        in_channels: Union[List, int, None] = None,
        out_channels: Union[List, int, None] = None,
        kernel_size: Union[List, Tuple, None] = None,
        stride: Union[List, Tuple, None] = None,
        padding: Union[List, Tuple, None] = None,
        dilation: Union[List, Tuple, None] = None,
        groups: Union[List, int, None] = None,
        bias: Union[List, bool, None] = None,
        padding_mode: Union[List, str, None] = None,
        batches=1,
        dtype=torch.float32,
    ):
        super().__init__()
        self.nbr_convs = nbr_conv

        # Handle default values
        in_channels = [2] * nbr_conv if in_channels is None else in_channels
        out_channels = [1 * nbr_conv] if out_channels is None else out_channels
        kernel_size = [3] * nbr_conv if kernel_size is None else kernel_size
        stride = [2] * nbr_conv if stride is None else stride
        padding = [1] * nbr_conv if padding is None else padding
        dilation = [1] * nbr_conv if dilation is None else dilation
        groups = [1] * nbr_conv if groups is None else groups
        bias = [True] * nbr_conv if bias is None else bias
        padding_mode = ["zeros"] * nbr_conv if padding_mode is None else padding_mode

        # This allows the input parameters to be either a single value or a list
        # as type hint implies
        if not isinstance(in_channels, List):
            in_channels = [in_channels]
        if not isinstance(out_channels, List):
            out_channels = [out_channels]
        if not isinstance(kernel_size, List):
            kernel_size = [kernel_size]
        if not isinstance(stride, List):
            stride = [stride]
        if not isinstance(padding, List):
            padding = [padding]
        if not isinstance(dilation, List):
            dilation = [dilation]
        if not isinstance(groups, List):
            groups = [groups]
        if not isinstance(bias, List):
            bias = [bias]
        if not isinstance(padding_mode, List):
            padding_mode = [padding_mode]

        # Generate test data if not provided
        if inputs is None:
            self.inputs = (torch.randn(batches, in_channels[0], length).to(dtype),)
        else:
            self.inputs = (inputs,)

        # Build chain of convs
        for i in range(self.nbr_convs):
            setattr(
                self,
                f"conv_{i}",
                torch.nn.Conv1d(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    padding=padding[i],
                    dilation=dilation[i],
                    groups=groups[i],
                    bias=bias[i],
                    padding_mode=padding_mode[i],
                ).to(dtype),
            )

    def get_inputs(self):
        return self.inputs

    def forward(self, x):
        for i in range(self.nbr_convs):
            conv = getattr(self, f"conv_{i}")
            x = conv(x)
        return x


conv1d_2_3x2x40_nobias = Conv1d(
    in_channels=2,
    out_channels=3,
    kernel_size=2,
    stride=1,
    bias=False,
    padding=0,
    length=40,
    batches=1,
)

conv1d_3_1x3x256_st1 = Conv1d(
    in_channels=3,
    out_channels=10,
    kernel_size=3,
    stride=1,
    padding=0,
    length=256,
    batches=1,
)

conv1d_3_1x3x12_st2_pd1 = Conv1d(
    in_channels=3,
    out_channels=4,
    kernel_size=3,
    stride=2,
    padding=1,
    length=12,
    batches=1,
)

conv1d_1_1x2x128_st1 = Conv1d(
    in_channels=2,
    out_channels=1,
    kernel_size=1,
    stride=1,
    padding=0,
    length=128,
    batches=1,
)

conv1d_2_1x2x14_st2 = Conv1d(
    in_channels=2,
    out_channels=1,
    kernel_size=2,
    stride=2,
    padding=0,
    length=14,
    batches=1,
)

conv1d_5_3x2x128_st1 = Conv1d(
    in_channels=2,
    out_channels=3,
    kernel_size=5,
    stride=1,
    padding=0,
    length=128,
    batches=3,
)

conv1d_3_1x3x224_st2_pd1 = Conv1d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=2,
    padding=1,
    length=224,
    batches=1,
)

two_conv1d_nobias = Conv1d(
    nbr_conv=2,
    length=256,
    in_channels=[3, 10],
    out_channels=[10, 15],
    kernel_size=[5, 5],
    stride=[1, 1],
    padding=[0, 0],
    bias=[False, False],
    batches=1,
)

two_conv1d = Conv1d(
    nbr_conv=2,
    length=256,
    in_channels=[3, 10],
    out_channels=[10, 15],
    kernel_size=[5, 5],
    stride=[1, 1],
    padding=[0, 0],
    bias=[True, True],
    batches=1,
)

# Shenanigan to get a nicer output when test fails. With unittest it looks like:
# FAIL: test_conv1d_tosa_BI_2_3x3_1x3x12x12_st2_pd1
testsuite = [
    ("2_3x2x40_nobias", conv1d_2_3x2x40_nobias),
    ("3_1x3x256_st1", conv1d_3_1x3x256_st1),
    ("3_1x3x12_st2_pd1", conv1d_3_1x3x12_st2_pd1),
    ("1_1x2x128_st1", conv1d_1_1x2x128_st1),
    ("2_1x2x14_st2", conv1d_2_1x2x14_st2),
    ("5_3x2x128_st1", conv1d_5_3x2x128_st1),
    ("3_1x3x224_st2_pd1", conv1d_3_1x3x224_st2_pd1),
    ("two_conv1d_nobias", two_conv1d_nobias),
    ("two_conv1d", two_conv1d),
]


class TestConv1D(unittest.TestCase):
    def _test_conv1d_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80.0+MI", permute_memory_to_nhwc=True
                ),
            )
            .export()
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_conv1d_tosa_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    "TOSA-0.80.0+BI", permute_memory_to_nhwc=True
                ),
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_conv1d_ethosu_BI_pipeline(
        self,
        module: torch.nn.Module,
        compile_spec: CompileSpec,
        test_data: Tuple[torch.Tensor],
    ):
        tester = (
            ArmTester(module, example_inputs=test_data, compile_spec=compile_spec)
            .quantize()
            .export()
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    @parameterized.expand(testsuite)
    def test_conv1d_tosa_MI(self, test_name, model):
        self._test_conv1d_tosa_MI_pipeline(model, model.get_inputs())

    @parameterized.expand(testsuite)
    def test_conv1d_tosa_BI(self, test_name, model):
        self._test_conv1d_tosa_BI_pipeline(model, model.get_inputs())

    # Expeted to fail as Conv1D requires transpoes which isn't supported on u55
    @parameterized.expand(testsuite)
    @unittest.expectedFailure
    def test_conv1d_u55_BI(self, test_name, model):
        self._test_conv1d_ethosu_BI_pipeline(
            model, common.get_u55_compile_spec(), model.get_inputs()
        )

    # This specific test case has numerical errors on FVP, MLETORCH-520.
    testsuite.remove(("5_3x2x128_st1", conv1d_5_3x2x128_st1))

    @parameterized.expand(testsuite)
    def test_conv1d_u85_BI(self, test_name, model):
        self._test_conv1d_ethosu_BI_pipeline(
            model, common.get_u85_compile_spec(), model.get_inputs()
        )
