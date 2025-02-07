# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple, Union

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.conv2d.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_convolution_default"


class Conv2d(torch.nn.Module):
    """
    Creates one or many chained 2D-convolutions. For multiple convolutions, the
    respective parameteres are provided as lists.
    """

    def __init__(
        self,
        height=8,
        width=8,
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
        dtype=torch.float,
    ):
        super().__init__()
        self.nbr_convs = nbr_conv

        # Handle default values
        in_channels = [2] * nbr_conv if in_channels is None else in_channels
        out_channels = [1 * nbr_conv] if out_channels is None else out_channels
        kernel_size = [(3, 3)] * nbr_conv if kernel_size is None else kernel_size
        stride = [(2, 2)] * nbr_conv if stride is None else stride
        padding = [(1, 1)] * nbr_conv if padding is None else padding
        dilation = [(1, 1)] * nbr_conv if dilation is None else dilation
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

        self.batches = batches
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.dtype = dtype

        # Build chain of convs
        for i in range(self.nbr_convs):
            setattr(
                self,
                f"conv_{i}",
                torch.nn.Conv2d(
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
        return (
            torch.randn(self.batches, self.in_channels[0], self.height, self.width).to(
                self.dtype
            ),
        )

    def forward(self, x):
        for i in range(self.nbr_convs):
            conv = getattr(self, f"conv_{i}")
            x = conv(x)
        return x


conv2d_2x2_3x2x40x40_nobias = Conv2d(
    in_channels=2,
    out_channels=3,
    kernel_size=(2, 2),
    stride=1,
    bias=False,
    padding=0,
    width=40,
    height=40,
    batches=3,
)

conv2d_3x3_1x3x256x256_st1 = Conv2d(
    in_channels=3,
    out_channels=10,
    kernel_size=(3, 3),
    stride=1,
    padding=0,
    width=256,
    height=256,
    batches=1,
)

conv2d_3x3_1x3x12x12_st2_pd1 = Conv2d(
    in_channels=3,
    out_channels=4,
    kernel_size=(3, 3),
    stride=2,
    padding=1,
    width=12,
    height=12,
    batches=1,
)

conv2d_1x1_1x2x128x128_st1 = Conv2d(
    in_channels=2,
    out_channels=1,
    kernel_size=(1, 1),
    stride=1,
    padding=0,
    width=128,
    height=128,
    batches=1,
)

conv2d_2x2_1x1x14x13_st2 = Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=(2, 2),
    stride=2,
    padding=0,
    width=14,
    height=13,
    batches=1,
)

conv2d_5x5_3x2x128x128_st1 = Conv2d(
    in_channels=2,
    out_channels=3,
    kernel_size=(5, 5),
    stride=1,
    padding=0,
    width=128,
    height=128,
    batches=3,
)

conv2d_3x3_1x3x224x224_st2_pd1 = Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=(3, 3),
    stride=2,
    padding=1,
    width=224,
    height=224,
    batches=1,
)

conv2d_5x5_1x3x14x15_st3_pd1 = Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=(5, 5),
    stride=3,
    padding=1,
    width=14,
    height=15,
    batches=1,
)

conv2d_7x7_1x3x16x16_st2_pd1_dl2 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(7, 7),
    stride=2,
    padding=1,
    dilation=2,
    width=16,
    height=16,
    batches=1,
)

conv2d_7x7_1x3x15x15_st1_pd0_dl1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(7, 7),
    stride=1,
    padding=0,
    dilation=1,
    width=15,
    height=15,
    batches=1,
)

conv2d_5x5_1x3x14x14_st5_pd0_dl1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(5, 5),
    stride=5,
    padding=0,
    dilation=1,
    width=14,
    height=14,
    batches=1,
)

conv2d_5x5_1x3x9x9_st5_pd0_dl1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(5, 5),
    stride=5,
    padding=0,
    dilation=1,
    width=9,
    height=9,
    batches=1,
)

conv2d_3x3_1x3x8x9_st3_pd0_dl1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3),
    stride=3,
    padding=0,
    dilation=1,
    width=8,
    height=9,
    batches=1,
)

conv2d_3x3_1x3x9x8_st3_pd0_dl1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3),
    stride=3,
    padding=0,
    dilation=1,
    width=8,
    height=9,
    batches=1,
)

conv2d_3x4_1x3x7x7_st3_pd0_dl1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 4),
    stride=3,
    padding=0,
    dilation=1,
    width=7,
    height=7,
    batches=1,
)

conv2d_4x3_1x3x7x7_st3_pd0_dl1 = Conv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=(4, 3),
    stride=3,
    padding=0,
    dilation=1,
    width=7,
    height=7,
    batches=1,
)

two_conv2d_nobias = Conv2d(
    nbr_conv=2,
    width=256,
    height=256,
    in_channels=[3, 10],
    out_channels=[10, 15],
    kernel_size=[(5, 5), (5, 5)],
    stride=[1, 1],
    padding=[0, 0],
    bias=[False, False],
    batches=1,
)

two_conv2d = Conv2d(
    nbr_conv=2,
    width=256,
    height=256,
    in_channels=[3, 10],
    out_channels=[10, 15],
    kernel_size=[(5, 5), (5, 5)],
    stride=[1, 1],
    padding=[0, 0],
    bias=[True, True],
    batches=1,
)

# Shenanigan to get a nicer output when test fails. With unittest it looks like:
# FAIL: test_conv2d_tosa_BI_2_3x3_1x3x12x12_st2_pd1
test_modules = {
    "2x2_3x2x40x40_nobias": conv2d_2x2_3x2x40x40_nobias,
    "3x3_1x3x256x256_st1": conv2d_3x3_1x3x256x256_st1,
    "3x3_1x3x12x12_st2_pd1": conv2d_3x3_1x3x12x12_st2_pd1,
    "1x1_1x2x128x128_st1": conv2d_1x1_1x2x128x128_st1,
    "2x2_1x1x14x13_st2_needs_adjust_pass": conv2d_2x2_1x1x14x13_st2,
    "5x5_1x3x14x15_st3_pd1_needs_adjust_pass": conv2d_5x5_1x3x14x15_st3_pd1,
    "7x7_1x3x16x16_st2_pd1_dl2_needs_adjust_pass": conv2d_7x7_1x3x16x16_st2_pd1_dl2,
    "7x7_1x3x15x15_st1_pd0_dl1_needs_adjust_pass": conv2d_7x7_1x3x15x15_st1_pd0_dl1,
    "5x5_1x3x14x14_st5_pd0_dl1_needs_adjust_pass": conv2d_5x5_1x3x14x14_st5_pd0_dl1,
    "5x5_1x3x9x9_st5_pd0_dl1_needs_adjust_pass": conv2d_5x5_1x3x9x9_st5_pd0_dl1,
    "3x3_1x3x9x8_st3_pd0_dl1_needs_adjust_pass": conv2d_3x3_1x3x9x8_st3_pd0_dl1,
    "3x3_1x3x8x9_st3_pd0_dl1_needs_adjust_pass": conv2d_3x3_1x3x8x9_st3_pd0_dl1,
    "3x4_1x3x7x7_st3_pd0_dl1_needs_adjust_pass": conv2d_3x4_1x3x7x7_st3_pd0_dl1,
    "4x3_1x3x7x7_st3_pd0_dl1_needs_adjust_pass": conv2d_4x3_1x3x7x7_st3_pd0_dl1,
    "5x5_3x2x128x128_st1": conv2d_5x5_3x2x128x128_st1,
    "3x3_1x3x224x224_st2_pd1": conv2d_3x3_1x3x224x224_st2_pd1,
    "two_conv2d_nobias": two_conv2d_nobias,
    "two_conv2d": two_conv2d,
}

fvp_xfails = {
    "2x2_3x2x40x40_nobias": "MLETORCH-520: Numerical issues on FVP.",
    "5x5_3x2x128x128_st1": "MLETORCH-520: Numerical issues on FVP.",
}
input_t = Tuple[torch.Tensor]


@common.parametrize("test_module", test_modules)
def test_conv2d_tosa_MI(test_module):
    pipeline = TosaPipelineMI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
def test_conv2d_tosa_BI(test_module):
    pipeline = TosaPipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op
    )
    pipeline.change_args("run_method_and_compare_outputs", qtol=1)
    pipeline.run()


@common.parametrize("test_module", test_modules)
def test_conv2d_u55_BI(test_module):
    pipeline = EthosU55PipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
def test_conv2d_u85_BI(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op, run_on_fvp=False
    )
    pipeline.run()


@common.parametrize("test_module", test_modules, fvp_xfails)
@common.SkipIfNoCorstone300
def test_conv2d_u55_BI_on_fvp(test_module):
    pipeline = EthosU55PipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


@common.parametrize("test_module", test_modules, fvp_xfails)
@common.SkipIfNoCorstone320
def test_conv2d_u85_BI_on_fvp(test_module):
    pipeline = EthosU85PipelineBI[input_t](
        test_module, test_module.get_inputs(), aten_op, exir_op, run_on_fvp=True
    )
    pipeline.run()


reject_suite = {
    "large_stride": Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(2, 4),
        stride=(2, 4),
        padding=1,
        width=10,
        height=14,
        batches=1,
    ),
    "large_kernel_height": Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(2, 65),
        stride=(1, 1),
        padding=0,
        width=70,
        height=70,
        batches=1,
    ),
    "large_kernel": Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=(70, 60),
        stride=(1,),
        padding=0,
        width=80,
        height=80,
        batches=1,
    ),
}


@common.parametrize("module", reject_suite)
def test_reject_conv2d_u55_BI(
    module: Conv2d,
):
    (
        ArmTester(
            module,
            example_inputs=module.get_inputs(),
            compile_spec=common.get_u55_compile_spec(),
        )
        .quantize()
        .export()
        .check_count({"torch.ops.aten.conv2d.default": 1})
        .check(["torch.ops.quantized_decomposed"])
        .to_edge_transform_and_lower()
        .check(["executorch_exir_dialects_edge__ops_aten_convolution_default"])
        .check_count({"torch.ops.higher_order.executorch_call_delegate": 0})
    )
