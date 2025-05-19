# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
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

    def __init__(self):
        super().__init__()
        # (t, c, n, s) = (6, 96, 1, 1)
        # 1. 1x1 CONV2d + ReLU6 (Pointwise)
        self.pointwise_conv2d = torch.nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=1, stride=1, groups=1
        )  ## (1, 128, 81, 81)
        self.batch_norm2d_16 = torch.nn.BatchNorm2d(128, affine=False)
        self.relu6 = torch.nn.ReLU6()

        # 2. 3x3 DepthwiseConv2d + ReLu6
        self.depthwise_conv2d = torch.nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=128,
        )  ## (1, 128, H, W)

        # 3. Linear 1x1 Conv2d
        self.pointwise_conv2d_linear = torch.nn.Conv2d(
            in_channels=128, out_channels=32, kernel_size=1, stride=1, groups=1
        )  ## (1, 32, 81, 81)

    def get_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.randn(1, 32, 81, 81),)

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

    test_data = {
        "combo_conv_relu_2_x_4d": lambda: (2 * torch.randn(1, 3, 256, 256),),
        "combo_conv_relu_0_5_x_4d": lambda: (0.5 * torch.randn(1, 3, 256, 256),),
        "combo_conv_relu_4d": lambda: (torch.randn(1, 3, 256, 256),),
        "combo_conv_relu_neg_0_5_x_4d": lambda: (-0.5 * torch.randn(1, 3, 256, 256),),
        "combo_conv_relu_neg_2_x_4d": lambda: (-2 * torch.randn(1, 3, 256, 256),),
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

    test_data = {
        "combo_conv_avgpool_20_x_4d": lambda: (20 * torch.randn(1, 3, 64, 32),),
        "combo_conv_avgpool_4d": lambda: (torch.randn(1, 3, 100, 200),),
        "combo_conv_avgpool_5_x_4d_randn": lambda: (5 * torch.randn(1, 3, 256, 256),),
        "combo_conv_avgpool_2_x_4d": lambda: (torch.rand(1, 3, 512, 128),),
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


def test_convolution_2d_tosa_MI_meandim():
    model = ComboConv2dMeandim()

    pipeline = TosaPipelineMI[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConv2dMeandim.edge_op_list,
    )
    pipeline.run()


def test_convolution_2d_tosa_BI_meandim():
    model = ComboConv2dMeandim()
    pipeline = TosaPipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConv2dMeandim.edge_op_list,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
def test_convolution_2d_u55_BI_meandim():
    model = ComboConv2dMeandim()
    pipeline = EthosU55PipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=ComboConv2dMeandim.edge_op_list,
        run_on_fvp=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_convolution_2d_u85_BI_meandim():
    model = ComboConv2dMeandim()
    pipeline = EthosU85PipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=ComboConv2dMeandim.edge_op_list,
        run_on_fvp=True,
    )
    pipeline.run()


##############################
## Conv + batch norm + relu ##
##############################
affine_params = {"affine": True, "_no_affine": False}


@common.parametrize("affine", affine_params)
def test_convolution_2d_tosa_MI_batchnorm_relu6(affine):
    model = ComboConvBatchnormRelu6(affine)
    pipeline = TosaPipelineMI[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConvBatchnormRelu6.edge_op_list,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
@common.parametrize("affine", affine_params)
def test_convolution_2d_tosa_BI_batchnorm_relu6(affine):
    model = ComboConvBatchnormRelu6(affine)
    pipeline = TosaPipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboConvBatchnormRelu6.edge_op_list,
    )
    pipeline.run()


@common.parametrize("affine", affine_params)
@common.XfailIfNoCorstone300
def test_convolution_2d_u55_BI_batchnorm_relu6(affine):
    model = ComboConvBatchnormRelu6(affine)
    pipeline = EthosU55PipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("affine", affine_params)
@common.XfailIfNoCorstone320
def test_convolution_2d_u85_BI_batchnorm_relu6(affine):
    model = ComboConvBatchnormRelu6(affine)
    pipeline = EthosU85PipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


##################
## Conv + ReLU6 ##
##################


@common.parametrize("test_data", ComboConvRelu6.test_data)
def test_convolution_2d_tosa_MI_relu6(test_data: torch.Tensor):
    model = ComboConvRelu6()
    pipeline = TosaPipelineMI[input_t1](
        model,
        test_data(),
        aten_op=[],
        exir_op=ComboConvRelu6.edge_op_list,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
@common.parametrize("test_data", ComboConvRelu6.test_data)
def test_convolution_2d_tosa_BI_relu6(test_data: torch.Tensor):
    model = ComboConvRelu6()
    pipeline = TosaPipelineBI[input_t1](
        model,
        test_data(),
        aten_op=[],
        exir_op=ComboConvRelu6.edge_op_list,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvRelu6.test_data)
@common.XfailIfNoCorstone300
def test_convolution_2d_u55_BI_relu6(test_data: torch.Tensor):
    model = ComboConvRelu6()
    pipeline = EthosU55PipelineBI[input_t1](
        model,
        test_data(),
        aten_ops=[],
        exir_ops=ComboConvRelu6.edge_op_list,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvRelu6.test_data)
@common.XfailIfNoCorstone320
def test_convolution_2d_u85_BI_relu6(test_data: torch.Tensor):
    model = ComboConvRelu6()
    pipeline = EthosU85PipelineBI[input_t1](
        model,
        test_data(),
        aten_ops=[],
        exir_ops=ComboConvRelu6.edge_op_list,
        run_on_fvp=True,
    )
    pipeline.run()


###############################
## Block bottleneck residual ##
###############################
def test_convolution_2d_tosa_MI_block_bottleneck():
    model = ComboBlockBottleneckResidual()
    pipeline = TosaPipelineMI[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboBlockBottleneckResidual.edge_op_list,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
def test_convolution_2d_tosa_BI_block_bottleneck():
    model = ComboBlockBottleneckResidual()
    pipeline = TosaPipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_op=[],
        exir_op=ComboBlockBottleneckResidual.edge_op_list,
    )
    pipeline.change_args("run_method_and_compare_outputs", model.get_inputs(), qtol=1)
    pipeline.run()


@common.XfailIfNoCorstone300
def test_convolution_2d_u55_BI_block_bottleneck():
    model = ComboBlockBottleneckResidual()
    pipeline = EthosU55PipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
def test_convolution_2d_u85_BI_block_bottleneck():
    model = ComboBlockBottleneckResidual()
    pipeline = EthosU85PipelineBI[input_t1](
        model,
        model.get_inputs(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


######################
## Conv + AvgPool2d ##
######################


@common.parametrize("test_data", ComboConvAvgPool2d.test_data)
def test_convolution_2d_tosa_MI_avgpool2d(test_data: torch.Tensor):
    model = ComboConvAvgPool2d()
    pipeline = TosaPipelineMI[input_t1](
        model,
        test_data(),
        aten_op=[],
        exir_op=ComboConvAvgPool2d.edge_op_list,
    )
    pipeline.run()


@pytest.mark.flaky(reruns=5)  # TODO: Investigate flakyness (MLTORCH-307)
@common.parametrize("test_data", ComboConvAvgPool2d.test_data)
def test_convolution_2d_tosa_BI_avgpool2d(test_data: torch.Tensor):
    model = ComboConvAvgPool2d()
    pipeline = TosaPipelineBI[input_t1](
        model,
        test_data(),
        aten_op=[],
        exir_op=ComboConvAvgPool2d.edge_op_list,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvAvgPool2d.test_data)
@common.XfailIfNoCorstone300
def test_convolution_2d_u55_BI_avgpool2d(test_data: torch.Tensor):
    model = ComboConvAvgPool2d()
    pipeline = EthosU55PipelineBI[input_t1](
        model,
        test_data(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", ComboConvAvgPool2d.test_data)
@common.XfailIfNoCorstone320
def test_convolution_2d_u85_BI_avgpool2d(test_data: torch.Tensor):
    model = ComboConvAvgPool2d()
    pipeline = EthosU85PipelineBI[input_t1](
        model,
        test_data(),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()
