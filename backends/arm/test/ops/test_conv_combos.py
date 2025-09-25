# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t1 = Tuple[torch.Tensor]

from torch.nn.parameter import Parameter


"""
This file contain unit tests where conv are combined with other ops.
"""


class ComboBlockBottleneckResidual(torch.nn.Module):
    # This is the essence of MobileNetV2. Ref: https://arxiv.org/abs/1801.04381
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    ]

    test_data_INT = {
        "per_channel_quant=True": True,
        "per_channel_quant=False": False,
    }

    def __init__(self):
        super().__init__()
        # (t, c, n, s) = (6, 96, 1, 1)
        # 1. 1x1 CONV2d + ReLU6 (Pointwise)
        self.pointwise_conv2d = torch.nn.Conv2d(
            in_channels=16, out_channels=96, kernel_size=1, stride=1, groups=1
        )  ## (1, 128, 81, 81)
        self.batch_norm2d_16 = torch.nn.BatchNorm2d(96, affine=False)
        self.relu6 = torch.nn.ReLU6()

        # 2. 3x3 DepthwiseConv2d + ReLu6
        self.depthwise_conv2d = torch.nn.Conv2d(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=96,
        )  ## (1, 128, H, W)

        # 3. Linear 1x1 Conv2d
        self.pointwise_conv2d_linear = torch.nn.Conv2d(
            in_channels=96, out_channels=16, kernel_size=1, stride=1, groups=1
        )  ## (1, 32, 81, 81)

    def get_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.randn(1, 16, 81, 81),)

    def forward(self, x):
        input = x
        # 1x1 CONV2d + ReLU6 (Pointwise)
        x = self.pointwise_conv2d(x)
        x = self.batch_norm2d_16(x)
        x = self.relu6(x)

        # 3x3 DepthwiseConv2d + ReLu6
        x = self.depthwise_conv2d(x)
        x = self.batch_norm2d_16(x)
        x = self.relu6(x)

        # Linear 1x1 Conv2d
        x = self.pointwise_conv2d_linear(x)

        # Final Residual Connection
        x = x + input

        return x


class ComboConv2dMeandim(torch.nn.Module):
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten_mean_dim",
    ]

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=5, stride=1, bias=False
        )
        # will be specialized to aten.mean.dim
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

    def get_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.randn(1, 3, 128, 128),)

    def forward(self, x):
        x = self.conv2d(x)
        return self.adaptive_avg_pool2d(x)


class ComboConvBatchnormRelu6(torch.nn.Module):
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default",
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
    ]

    test_data_FP = {
        "affine=True": True,
        "affine=False": False,
    }

    test_data_INT = {
        "affine=True,per_channel_quant=True": (True, True),
        "affine=True,per_channel_quant=False": (True, False),
        "affine=False,per_channel_quant=True": (False, True),
        "affine=False,per_channel_quant=False": (False, False),
    }

    def __init__(self, affine: bool):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.batch_norm2d = torch.nn.BatchNorm2d(3, affine=affine)
        self.batch_norm2d.running_mean = torch.rand(3)
        self.batch_norm2d.running_var = torch.rand(3)
        self.batch_norm2d.weight = Parameter(torch.rand(3))
        self.batch_norm2d.bias = Parameter(torch.rand(3))
        self.relu6 = torch.nn.ReLU6()

    def get_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.randn(1, 3, 256, 256),)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm2d(x)
        x = self.relu6(x)
        return x


class ComboConvRelu6(torch.nn.Module):
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
    ]

    test_data_FP = {
        "combo_conv_relu_2_x_4d": lambda: (2 * torch.randn(1, 3, 256, 256),),
        "combo_conv_relu_0_5_x_4d": lambda: (0.5 * torch.randn(1, 3, 256, 256),),
        "combo_conv_relu_4d": lambda: (torch.randn(1, 3, 256, 256),),
        "combo_conv_relu_neg_0_5_x_4d": lambda: (-0.5 * torch.randn(1, 3, 256, 256),),
        "combo_conv_relu_neg_2_x_4d": lambda: (-2 * torch.randn(1, 3, 256, 256),),
    }

    # Generate a new test set paired with per_channel_quant=True/False.
    test_data_INT = {
        # test_name: (input, per_channel_quant)
        f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
        for (k, v) in test_data_FP.items()
        for q in [True, False]
    }

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.relu6 = torch.nn.ReLU6()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu6(x)
        return x


class ComboConvAvgPool2d(torch.nn.Module):
    edge_op_list = [
        "executorch_exir_dialects_edge__ops_aten_convolution_default",
        "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default",
    ]

    test_data_FP = {
        "combo_conv_avgpool_20_x_4d": lambda: (20 * torch.randn(1, 3, 64, 32),),
        "combo_conv_avgpool_4d": lambda: (torch.randn(1, 3, 100, 200),),
        "combo_conv_avgpool_5_x_4d_randn": lambda: (5 * torch.randn(1, 3, 256, 256),),
        "combo_conv_avgpool_2_x_4d": lambda: (torch.rand(1, 3, 512, 128),),
    }

    # Generate a new test set paired with per_channel_quant=True/False.
    test_data_INT = {
        # test_name: (input, per_channel_quant)
        f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
        for (k, v) in test_data_FP.items()
        for q in [True, False]
    }

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.conv2d(x)
        x = self.avg_pool2d(x)
        return x


####################
## Conv + meandim ##
####################


def test_convolution_2d_tosa_FP_meandim():
    model = ComboConv2dMeandim()
    pipeline = TosaPipelineFP[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConv2dMeandim.edge_op_list,
    )
    pipeline.run()


def test_convolution_2d_tosa_INT_meandim():
    model = ComboConv2dMeandim()
    pipeline = TosaPipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConv2dMeandim.edge_op_list,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_convolution_2d_u55_INT_meandim():
    model = ComboConv2dMeandim()
    pipeline = EthosU55PipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=ComboConv2dMeandim.edge_op_list,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_convolution_2d_u85_INT_meandim():
    model = ComboConv2dMeandim()
    pipeline = EthosU85PipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=ComboConv2dMeandim.edge_op_list,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_FP_meandim():
    model = ComboConv2dMeandim()
    pipeline = VgfPipeline[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConv2dMeandim.edge_op_list,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_INT_meandim():
    model = ComboConv2dMeandim()
    pipeline = VgfPipeline[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConv2dMeandim.edge_op_list,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


##############################
## Conv + batch norm + relu ##
##############################


@common.parametrize("test_data", ComboConvBatchnormRelu6.test_data_FP)
def test_convolution_2d_tosa_FP_batchnorm_relu6(test_data):
    affine = test_data
    model = ComboConvBatchnormRelu6(affine)
    pipeline = TosaPipelineFP[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConvBatchnormRelu6.edge_op_list,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
@common.parametrize("test_data", ComboConvBatchnormRelu6.test_data_INT)
def test_convolution_2d_tosa_INT_batchnorm_relu6(test_data):
    affine, per_channel_quantization = test_data
    model = ComboConvBatchnormRelu6(affine)
    pipeline = TosaPipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConvBatchnormRelu6.edge_op_list,
        per_channel_quantization=per_channel_quantization,
        qtol=1,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvBatchnormRelu6.test_data_INT)
@common.XfailIfNoCorstone300
def test_convolution_2d_u55_INT_batchnorm_relu6(test_data):
    affine, per_channel_quantization = test_data
    model = ComboConvBatchnormRelu6(affine)
    pipeline = EthosU55PipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvBatchnormRelu6.test_data_INT)
@common.XfailIfNoCorstone320
def test_convolution_2d_u85_INT_batchnorm_relu6(test_data):
    affine, per_channel_quantization = test_data
    model = ComboConvBatchnormRelu6(affine)
    pipeline = EthosU85PipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvBatchnormRelu6.test_data_FP)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_FP_batchnorm_relu6(test_data):
    affine = test_data
    model = ComboConvBatchnormRelu6(affine)
    pipeline = VgfPipeline[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConvBatchnormRelu6.edge_op_list,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvBatchnormRelu6.test_data_INT)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_INT_batchnorm_relu6(test_data):
    affine, per_channel_quantization = test_data
    model = ComboConvBatchnormRelu6(affine)
    pipeline = VgfPipeline[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConvBatchnormRelu6.edge_op_list,
        tosa_version="TOSA-1.0+INT",
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


##################
## Conv + ReLU6 ##
##################


@common.parametrize("test_data", ComboConvRelu6.test_data_FP)
def test_convolution_2d_tosa_FP_relu6(test_data):
    model = ComboConvRelu6()
    pipeline = TosaPipelineFP[input_t1](
        model,
        test_data(),
        aten_op=[],
        exir_op=ComboConvRelu6.edge_op_list,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
@common.parametrize("test_data", ComboConvRelu6.test_data_INT)
def test_convolution_2d_tosa_INT_relu6(test_data):
    input, per_channel_quantization = test_data()
    model = ComboConvRelu6()
    pipeline = TosaPipelineINT[input_t1](
        model,
        input,
        aten_op=[],
        exir_op=ComboConvRelu6.edge_op_list,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvRelu6.test_data_INT)
@common.XfailIfNoCorstone300
def test_convolution_2d_u55_INT_relu6(test_data):
    input, per_channel_quantization = test_data()
    model = ComboConvRelu6()
    pipeline = EthosU55PipelineINT[input_t1](
        model,
        input,
        aten_ops=[],
        exir_ops=ComboConvRelu6.edge_op_list,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvRelu6.test_data_INT)
@common.XfailIfNoCorstone320
def test_convolution_2d_u85_INT_relu6(test_data):
    input, per_channel_quantization = test_data()
    model = ComboConvRelu6()
    pipeline = EthosU85PipelineINT[input_t1](
        model,
        input,
        aten_ops=[],
        exir_ops=ComboConvRelu6.edge_op_list,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvRelu6.test_data_FP)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_FP_relu6(test_data):
    model = ComboConvRelu6()
    pipeline = VgfPipeline[input_t1](
        model,
        test_data(),
        aten_op=[],
        exir_op=ComboConvRelu6.edge_op_list,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvRelu6.test_data_INT)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_INT_relu6(test_data):
    input, per_channel_quantization = test_data()
    model = ComboConvRelu6()
    pipeline = VgfPipeline[input_t1](
        model,
        input,
        aten_op=[],
        exir_op=ComboConvRelu6.edge_op_list,
        tosa_version="TOSA-1.0+INT",
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


###############################
## Block bottleneck residual ##
###############################
def test_convolution_2d_tosa_FP_block_bottleneck():
    model = ComboBlockBottleneckResidual()
    pipeline = TosaPipelineFP[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboBlockBottleneckResidual.edge_op_list,
    )
    pipeline.run()


@common.parametrize("test_data", ComboBlockBottleneckResidual.test_data_INT)
@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
def test_convolution_2d_tosa_INT_block_bottleneck(test_data):
    per_channel_quantization = test_data
    model = ComboBlockBottleneckResidual()
    pipeline = TosaPipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboBlockBottleneckResidual.edge_op_list,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.change_args("run_method_and_compare_outputs", model.get_inputs(), qtol=1)
    pipeline.run()


@common.parametrize("test_data", ComboBlockBottleneckResidual.test_data_INT)
@common.XfailIfNoCorstone300
def test_convolution_2d_u55_INT_block_bottleneck(test_data):
    per_channel_quantization = test_data
    model = ComboBlockBottleneckResidual()
    pipeline = EthosU55PipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboBlockBottleneckResidual.test_data_INT)
@common.XfailIfNoCorstone320
def test_convolution_2d_u85_INT_block_bottleneck(test_data):
    per_channel_quantization = test_data
    model = ComboBlockBottleneckResidual()
    pipeline = EthosU85PipelineINT[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_FP_block_bottleneck():
    model = ComboBlockBottleneckResidual()
    pipeline = VgfPipeline[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboBlockBottleneckResidual.edge_op_list,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", ComboBlockBottleneckResidual.test_data_INT)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_INT_block_bottleneck(test_data):
    per_channel_quantization = test_data
    model = ComboBlockBottleneckResidual()
    pipeline = VgfPipeline[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboBlockBottleneckResidual.edge_op_list,
        tosa_version="TOSA-1.0+INT",
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


######################
## Conv + AvgPool2d ##
######################


@common.parametrize("test_data", ComboConvAvgPool2d.test_data_FP)
def test_convolution_2d_tosa_FP_avgpool2d(test_data):
    model = ComboConvAvgPool2d()
    pipeline = TosaPipelineFP[input_t1](
        model,
        test_data(),
        aten_op=[],
        exir_op=ComboConvAvgPool2d.edge_op_list,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
@common.parametrize("test_data", ComboConvAvgPool2d.test_data_INT)
def test_convolution_2d_tosa_INT_avgpool2d(test_data):
    input, per_channel_quantization = test_data()
    model = ComboConvAvgPool2d()
    pipeline = TosaPipelineINT[input_t1](
        model,
        input,
        aten_op=[],
        exir_op=ComboConvAvgPool2d.edge_op_list,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvAvgPool2d.test_data_INT)
@common.XfailIfNoCorstone300
def test_convolution_2d_u55_INT_avgpool2d(test_data):
    input, per_channel_quantization = test_data()
    model = ComboConvAvgPool2d()
    pipeline = EthosU55PipelineINT[input_t1](
        model,
        input,
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvAvgPool2d.test_data_INT)
@common.XfailIfNoCorstone320
def test_convolution_2d_u85_INT_avgpool2d(test_data):
    input, per_channel_quantization = test_data()
    model = ComboConvAvgPool2d()
    pipeline = EthosU85PipelineINT[input_t1](
        model,
        input,
        aten_ops=[],
        exir_ops=[],
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvAvgPool2d.test_data_FP)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_FP_avgpool2d(test_data):
    model = ComboConvAvgPool2d()
    pipeline = VgfPipeline[input_t1](
        model,
        test_data(),
        aten_op=[],
        exir_op=ComboConvAvgPool2d.edge_op_list,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvAvgPool2d.test_data_INT)
@common.SkipIfNoModelConverter
def test_convolution_2d_vgf_INT_avgpool2d(test_data):
    input, per_channel_quantization = test_data()
    model = ComboConvAvgPool2d()
    pipeline = VgfPipeline[input_t1](
        model,
        input,
        aten_op=[],
        exir_op=ComboConvAvgPool2d.edge_op_list,
        tosa_version="TOSA-1.0+INT",
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()
