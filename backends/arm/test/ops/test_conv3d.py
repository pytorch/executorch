# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple, Union

import pytest
import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a8w4_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.conv3d.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_convolution_default"


class Conv3d(torch.nn.Module):
    """
    Creates one or many chained 3D-convolutions. For multiple convolutions, the
    respective parameteres are provided as lists.
    """

    def __init__(
        self,
        height=8,
        width=8,
        depth=8,
        nbr_conv=1,  # Number of chained convs
        in_channels: Union[List, int, None] = None,
        out_channels: Union[List, int, None] = None,
        kernel_size: Union[List, Tuple, None] = None,
        stride: Union[List, Tuple, int, None] = None,
        padding: Union[List, Tuple, int, None] = None,
        dilation: Union[List, Tuple, int, None] = None,
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
        kernel_size = [(3, 3, 1)] * nbr_conv if kernel_size is None else kernel_size
        stride = [(2, 2, 1)] * nbr_conv if stride is None else stride
        padding = [(1, 1, 1)] * nbr_conv if padding is None else padding
        dilation = [(1, 1, 1)] * nbr_conv if dilation is None else dilation
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
        self.depth = depth
        self.dtype = dtype

        # Build chain of convs
        for i in range(self.nbr_convs):
            setattr(
                self,
                f"conv_{i}",
                torch.nn.Conv3d(
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
            torch.randn(
                self.batches,
                self.in_channels[0],
                self.depth,
                self.height,
                self.width,
            ).to(self.dtype),
        )

    def forward(self, x):
        for i in range(self.nbr_convs):
            conv = getattr(self, f"conv_{i}")
            x = conv(x)
        return x


class Conv3dMultiOp(torch.nn.Module):
    """
    Mixed Conv3d/Conv2d pipeline used to verify spatial-rank propagation across ops.

    Topology:
        conv3d -> reshape -> conv2d -> reshape/permutation -> conv2d -> reshape -> add(5D)
    """

    def __init__(self, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.conv3d = torch.nn.Conv3d(
            in_channels=2,
            out_channels=4,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
        ).to(dtype)
        self.conv2d_main = torch.nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        ).to(dtype)
        self.conv2d_pointwise = torch.nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=1,
            stride=1,
            padding=0,
        ).to(dtype)
        self.activation = torch.nn.ReLU()

    def get_inputs(self):
        return (torch.randn(1, 2, 3, 8, 8).to(self.dtype),)

    def forward(self, x):
        x3d = self.conv3d(x)
        batches, channels, depth, height, width = x3d.shape

        reshaped = x3d.reshape(batches * depth, channels, height, width)
        conv2d_out = self.activation(self.conv2d_main(reshaped))

        conv2d_out_5d = (
            conv2d_out.reshape(batches, depth, channels, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        reshaped_again = conv2d_out_5d.permute(0, 2, 1, 3, 4).reshape(
            batches * depth, channels, height, width
        )
        conv2d_pointwise_out = self.conv2d_pointwise(reshaped_again)
        conv2d_pointwise_out_5d = (
            conv2d_pointwise_out.reshape(batches, depth, channels, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

        return conv2d_pointwise_out_5d + x3d


class DepthwiseConv3d(torch.nn.Module):
    def __init__(self, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.conv = torch.nn.Conv3d(
            in_channels=2,
            out_channels=4,
            kernel_size=(3, 3, 3),
            padding=1,
            groups=2,
        ).to(dtype)

    def get_inputs(self):
        return (torch.randn(1, 2, 3, 8, 8).to(self.dtype),)

    def forward(self, x):
        return self.conv(x)


conv3d_2x2_3x2x14x14_nobias = Conv3d(
    in_channels=2,
    out_channels=3,
    kernel_size=(2, 2, 2),
    stride=1,
    bias=False,
    padding=0,
    width=14,
    height=14,
    batches=2,
)

conv3d_3x3_1x3x24x24_st1 = Conv3d(
    in_channels=3,
    out_channels=10,
    kernel_size=(3, 3, 3),
    stride=1,
    padding=0,
    width=24,
    height=24,
    batches=1,
)

conv3d_3x3_1x3x12x12_st2_pd1 = Conv3d(
    in_channels=3,
    out_channels=4,
    kernel_size=(3, 3, 3),
    stride=2,
    padding=1,
    width=12,
    height=12,
    batches=1,
)

conv3d_1x1_1x2x16x16_st1 = Conv3d(
    in_channels=2,
    out_channels=1,
    kernel_size=(1, 1, 1),
    stride=1,
    padding=0,
    width=16,
    height=16,
    batches=1,
)

conv3d_2x2_1x1x14x13_st2 = Conv3d(
    in_channels=1,
    out_channels=1,
    kernel_size=(2, 2, 2),
    stride=2,
    padding=0,
    width=14,
    height=13,
    batches=1,
)

conv3d_5x5_3x2x24x24_st1 = Conv3d(
    in_channels=2,
    out_channels=3,
    kernel_size=(5, 5, 5),
    stride=1,
    padding=0,
    width=24,
    height=24,
    batches=2,
)

conv3d_3x3_1x3x28x28_st2_pd1 = Conv3d(
    in_channels=3,
    out_channels=16,
    kernel_size=(3, 3, 3),
    stride=2,
    padding=1,
    width=28,
    height=28,
    batches=1,
)

conv3d_5x5_1x3x14x15_st3_pd1 = Conv3d(
    in_channels=3,
    out_channels=16,
    kernel_size=(5, 5, 5),
    stride=3,
    padding=1,
    width=14,
    height=15,
    batches=1,
)

conv3d_7x7_1x3x16x16_st2_pd1_dl2 = Conv3d(
    in_channels=3,
    out_channels=3,
    kernel_size=(7, 7, 7),
    stride=2,
    padding=3,
    dilation=1,
    width=16,
    height=16,
    batches=1,
)

conv3d_7x7_1x3x15x15_st1_pd0_dl1 = Conv3d(
    in_channels=3,
    out_channels=3,
    kernel_size=(7, 7, 7),
    stride=1,
    padding=0,
    dilation=1,
    width=15,
    height=15,
    batches=1,
)

conv3d_5x5_1x3x14x14_st5_pd0_dl1 = Conv3d(
    in_channels=3,
    out_channels=3,
    kernel_size=(5, 5, 5),
    stride=5,
    padding=0,
    dilation=1,
    width=14,
    height=14,
    batches=1,
)

conv3d_5x5_1x3x9x9_st5_pd0_dl1 = Conv3d(
    in_channels=3,
    out_channels=3,
    kernel_size=(5, 5, 5),
    stride=5,
    padding=0,
    dilation=1,
    width=9,
    height=9,
    batches=1,
)

conv3d_3x3_1x3x8x9_st3_pd0_dl1 = Conv3d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3, 3),
    stride=3,
    padding=0,
    dilation=1,
    width=8,
    height=9,
    batches=1,
)

conv3d_3x3_1x3x9x8_st3_pd0_dl1 = Conv3d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 3, 3),
    stride=3,
    padding=0,
    dilation=1,
    width=8,
    height=9,
    batches=1,
)

conv3d_3x4_1x3x7x7_st3_pd0_dl1 = Conv3d(
    in_channels=3,
    out_channels=3,
    kernel_size=(3, 4, 3),
    stride=3,
    padding=0,
    dilation=1,
    width=7,
    height=7,
    batches=1,
)

conv3d_4x3_1x3x7x7_st3_pd0_dl1 = Conv3d(
    in_channels=3,
    out_channels=3,
    kernel_size=(4, 3, 3),
    stride=3,
    padding=0,
    dilation=1,
    width=7,
    height=7,
    batches=1,
)

test_data_FP = {
    "2x2_3x2x14x14_nobias": lambda: conv3d_2x2_3x2x14x14_nobias,
    "3x3_1x3x24x24_st1": lambda: conv3d_3x3_1x3x24x24_st1,
    "3x3_1x3x12x12_st2_pd1": lambda: conv3d_3x3_1x3x12x12_st2_pd1,
    "1x1_1x2x16x16_st1": lambda: conv3d_1x1_1x2x16x16_st1,
    "2x2_1x1x14x13_st2_needs_adjust_pass": lambda: conv3d_2x2_1x1x14x13_st2,
    "5x5_1x3x14x15_st3_pd1_needs_adjust_pass": lambda: conv3d_5x5_1x3x14x15_st3_pd1,
    "7x7_1x3x16x16_st2_pd1_dl2_needs_adjust_pass": lambda: conv3d_7x7_1x3x16x16_st2_pd1_dl2,
    "7x7_1x3x15x15_st1_pd0_dl1_needs_adjust_pass": lambda: conv3d_7x7_1x3x15x15_st1_pd0_dl1,
    "5x5_1x3x14x14_st5_pd0_dl1_needs_adjust_pass": lambda: conv3d_5x5_1x3x14x14_st5_pd0_dl1,
    "5x5_1x3x9x9_st5_pd0_dl1_needs_adjust_pass": lambda: conv3d_5x5_1x3x9x9_st5_pd0_dl1,
    "3x3_1x3x9x8_st3_pd0_dl1_needs_adjust_pass": lambda: conv3d_3x3_1x3x9x8_st3_pd0_dl1,
    "3x3_1x3x8x9_st3_pd0_dl1_needs_adjust_pass": lambda: conv3d_3x3_1x3x8x9_st3_pd0_dl1,
    "3x4_1x3x7x7_st3_pd0_dl1_needs_adjust_pass": lambda: conv3d_3x4_1x3x7x7_st3_pd0_dl1,
    "4x3_1x3x7x7_st3_pd0_dl1_needs_adjust_pass": lambda: conv3d_4x3_1x3x7x7_st3_pd0_dl1,
    "5x5_3x2x24x24_st1": lambda: conv3d_5x5_3x2x24x24_st1,
    "3x3_1x3x28x28_st2_pd1": lambda: conv3d_3x3_1x3x28x28_st2_pd1,
}

test_data_FP_bf16 = {
    "bf16_3x3": lambda: Conv3d(
        height=10,
        width=10,
        depth=6,
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        bias=True,
        dtype=torch.bfloat16,
    ),
    "bf16_1x1": lambda: Conv3d(
        height=6,
        width=6,
        depth=4,
        in_channels=2,
        out_channels=2,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        bias=False,
        dtype=torch.bfloat16,
    ),
}

# Generate a new test set paired with per_channel_quant=True/False.
test_data_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_FP.items()
    for q in [True, False]
}

test_data_INT16 = {
    f"{k},16a8w,per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_FP.items()
    for q in [True, False]
}


def _get_dtype_count(model: torch.nn.Module):
    nbr_convs: int = model.nbr_convs  # noqa
    return {
        "CONST": {"INT4": nbr_convs * 2},
        "CONV3D": {"INT32": nbr_convs},
        "RESCALE": {"INT8": nbr_convs},
    }


input_t = Tuple[torch.Tensor]


@common.parametrize("test_data", test_data_FP | test_data_FP_bf16)
def test_convolution_3d_tosa_FP(test_data):
    pipeline = TosaPipelineFP[input_t](
        test_data(),
        test_data().get_inputs(),
        aten_op,
        exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
def test_convolution_3d_tosa_INT(test_data):
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
def test_convolution_3d_tosa_INT_a8w4(test_data):
    model, per_channel_quantization = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        tosa_extensions=["int4"],
        qtol=1,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check_dtype_count,
        _get_dtype_count(model),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT16)
def test_convolution_3d_tosa_INT_a16w8(test_data):
    model, per_channel_quantization = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
        qtol=1,
    )
    pipeline.run()


def test_convolution_3d_tosa_FP_multi_op():
    """Ensure mixed Conv3d/Conv2d graphs keep correct spatial annotations."""
    model = Conv3dMultiOp()
    pipeline = TosaPipelineFP[input_t](model, model.get_inputs(), aten_op, exir_op)
    pipeline.run()


def test_convolution_3d_tosa_INT_multi_op():
    """Ensure mixed Conv3d/Conv2d graphs keep correct spatial annotations."""
    model = Conv3dMultiOp()
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
    )
    pipeline.run()


def test_convolution_3d_tosa_FP_depthwise():
    """Depthwise or Grouped Conv3d should be rejected until grouped support exists."""
    model = DepthwiseConv3d()
    pipeline = TosaPipelineFP[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        run_on_tosa_ref_model=False,
    )
    with pytest.raises(RuntimeError, match="CONV3D with groups != 1"):
        pipeline.run()


@common.parametrize("test_data", test_data_INT)
@pytest.mark.skip(reason="Ethos-U55 does not support CONV3D yet.")
def test_convolution_3d_u55_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@pytest.mark.skip(reason="Ethos-U55 does not support CONV3D yet.")
def test_convolution_3d_u55_INT_a8w4(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@pytest.mark.skip(reason="Ethos-U85 does not support CONV3D yet.")
def test_convolution_3d_u85_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@pytest.mark.skip(reason="Ethos-U85 does not support CONV3D yet.")
def test_convolution_3d_u85_INT_a8w4(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_FP)
@common.SkipIfNoModelConverter
def test_convolution_3d_vgf_no_quant(test_data):
    pipeline = VgfPipeline[input_t](
        test_data(),
        test_data().get_inputs(),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@common.SkipIfNoModelConverter
def test_convolution_3d_vgf_quant(test_data):
    model, per_channel_quantization = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_convolution_3d_vgf_no_quant_multi_op():
    """Ensure mixed Conv3d/Conv2d graphs keep correct spatial annotations."""
    model = Conv3dMultiOp()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_convolution_3d_vgf_quant_multi_op():
    """Ensure mixed Conv3d/Conv2d graphs keep correct spatial annotations."""
    model = Conv3dMultiOp()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


reject_suite = {
    "large_stride": lambda: Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(2, 2, 1),
        stride=(2, 4, 2),
        padding=1,
        width=10,
        height=14,
        batches=1,
    ),
    "large_kernel_z": lambda: Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(2, 2, 2),
        stride=1,
        padding=0,
        width=80,
        height=80,
        batches=1,
    ),
}


@common.parametrize("module", reject_suite)
def test_convolution_u55_INT_not_delegated_3d(module: Conv3d):
    OpNotSupportedPipeline(
        module(),
        module().get_inputs(),
        {"executorch_exir_dialects_edge__ops_aten_convolution_default": 1},
        quantize=True,
        u55_subset=True,
    ).run()
