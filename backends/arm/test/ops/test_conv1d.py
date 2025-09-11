# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple, Union

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.conv1d.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_convolution_default"

input_t = Tuple[torch.Tensor]


class Conv1d(torch.nn.Module):
    """
    Creates one or many chained 1D-convolutions. For multiple convolutions, the
    respective parameteres are provided as lists.
    """

    def __init__(
        self,
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

        self.batches = batches
        self.in_channels = in_channels
        self.length = length
        self.dtype = dtype

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
        return (
            torch.randn(self.batches, self.in_channels[0], self.length).to(self.dtype),
        )

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

conv1d_7_1x3x16_st2_pd1_dl2 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=7,
    stride=2,
    padding=1,
    dilation=2,
    length=16,
    batches=1,
)
conv1d_7_1x3x15_st1_pd0_dl1 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=7,
    stride=1,
    padding=0,
    dilation=1,
    length=15,
    batches=1,
)
conv1d_5_1x3x14_st5_pd0_dl1 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=5,
    stride=5,
    padding=0,
    dilation=1,
    length=14,
    batches=1,
)
conv1d_5_1x3x9_st5_pd0_dl1 = Conv1d(
    in_channels=3,
    out_channels=3,
    kernel_size=5,
    stride=5,
    padding=0,
    dilation=1,
    length=9,
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

test_data_FP = {
    "2_3x2x40_nobias": lambda: conv1d_2_3x2x40_nobias,
    "3_1x3x256_st1": lambda: conv1d_3_1x3x256_st1,
    "3_1x3x12_st2_pd1": lambda: conv1d_3_1x3x12_st2_pd1,
    "1_1x2x128_st1": lambda: conv1d_1_1x2x128_st1,
    "2_1x2x14_st2": lambda: conv1d_2_1x2x14_st2,
    "5_3x2x128_st1": lambda: conv1d_5_3x2x128_st1,
    "3_1x3x224_st2_pd1": lambda: conv1d_3_1x3x224_st2_pd1,
    "7_1x3x16_st2_pd1_dl2_needs_adjust_pass": lambda: conv1d_7_1x3x16_st2_pd1_dl2,
    "7_1x3x15_st1_pd0_dl1_needs_adjust_pass": lambda: conv1d_7_1x3x15_st1_pd0_dl1,
    "5_1x3x14_st5_pd0_dl1_needs_adjust_pass": lambda: conv1d_5_1x3x14_st5_pd0_dl1,
    "5_1x3x9_st5_pd0_dl1_needs_adjust_pass": lambda: conv1d_5_1x3x9_st5_pd0_dl1,
    "two_conv1d_nobias": lambda: two_conv1d_nobias,
    "two_conv1d": lambda: two_conv1d,
}

test_data_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_FP.items()
    for q in [True, False]
}


@common.parametrize("test_data", test_data_FP)
def test_convolution_1d_tosa_FP(test_data):
    pipeline = TosaPipelineFP[input_t](
        test_data(),
        test_data().get_inputs(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
def test_convolution_1d_tosa_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@common.XfailIfNoCorstone300
def test_convolution_1d_u55_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        run_on_fvp=True,
        per_channel_quantization=per_channel_quantization,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@common.XfailIfNoCorstone320
def test_convolution_1d_u85_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        run_on_fvp=True,
        per_channel_quantization=per_channel_quantization,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_FP)
@common.SkipIfNoModelConverter
def test_convolution_1d_vgf_FP(test_data):
    pipeline = VgfPipeline[input_t](
        test_data(),
        test_data().get_inputs(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@common.SkipIfNoModelConverter
def test_convolution_1d_vgf_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()
