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
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.exir.backend.backend_details import CompileSpec
from parameterized import parameterized

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
            in_channels=64, out_channels=384, kernel_size=1, stride=1, groups=1
        )  ## (1, 384, 81, 81)
        self.batch_norm2d_16 = torch.nn.BatchNorm2d(384, affine=False)
        self.relu6 = torch.nn.ReLU6()

        # 2. 3x3 DepthwiseConv2d + ReLu6
        self.depthwise_conv2d = torch.nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=384,
        )  ## (1, 384, H, W)

        # 3. Linear 1x1 Conv2d
        self.pointwise_conv2d_linear = torch.nn.Conv2d(
            in_channels=384, out_channels=64, kernel_size=1, stride=1, groups=1
        )  ## (1, 64, 81, 81)

    def get_inputs(self) -> Tuple[torch.Tensor]:
        return (torch.randn(1, 64, 81, 81),)

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

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, groups=1
        )
        self.batch_norm2d = torch.nn.BatchNorm2d(3, affine=False)
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

    test_data = [
        (20 * torch.randn(1, 3, 256, 256),),
        (5 * torch.randn(1, 3, 256, 256),),
        (torch.randn(1, 3, 256, 256),),
        (-5 * torch.randn(1, 3, 256, 256),),
    ]

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

    test_data = [
        (20 * torch.randn(1, 3, 64, 32),),
        (torch.randn(1, 3, 100, 200),),
        (5 * torch.randn(1, 3, 256, 256),),
        (torch.rand(1, 3, 512, 128),),
    ]

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


class TestConvCombos(unittest.TestCase):
    """Tests conv combined with other ops."""

    def _test_conv_combo_tosa_MI_pipeline(
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
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(list(module.edge_op_list))
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_conv_combo_tosa_BI_pipeline(
        self,
        module: torch.nn.Module,
        test_data: Tuple[torch.Tensor],
        atol: float = 1e-3,
        rtol: float = 1e-3,
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
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(list(module.edge_op_list))
            .to_executorch()
            .run_method_and_compare_outputs(
                inputs=test_data, atol=atol, rtol=rtol, qtol=1
            )
        )

    def _test_conv_combo_ethos_BI_pipeline(
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
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(list(module.edge_op_list))
            .to_executorch()
            .serialize()
        )
        if conftest.is_option_enabled("corstone_fvp"):
            tester.run_method_and_compare_outputs(qtol=1, inputs=test_data)

    ####################
    ## Conv + meandim ##
    ####################
    def test_conv_meandim_tosa_MI(self):
        model = ComboConv2dMeandim()
        self._test_conv_combo_tosa_MI_pipeline(model, model.get_inputs())

    def test_conv_meandim_tosa_BI(self):
        model = ComboConv2dMeandim()
        self._test_conv_combo_tosa_BI_pipeline(model, model.get_inputs())

    @pytest.mark.corstone_fvp
    def test_conv_meandim_u55_BI(self):
        model = ComboConv2dMeandim()
        self._test_conv_combo_ethos_BI_pipeline(
            model,
            common.get_u55_compile_spec(permute_memory_to_nhwc=True),
            model.get_inputs(),
        )

    @pytest.mark.corstone_fvp
    def test_conv_meandim_u85_BI(self):
        model = ComboConv2dMeandim()
        self._test_conv_combo_ethos_BI_pipeline(
            model,
            common.get_u85_compile_spec(permute_memory_to_nhwc=True),
            model.get_inputs(),
        )

    ##############################
    ## Conv + batch norm + relu ##
    ##############################
    def test_conv_batchnorm_relu6_tosa_MI(self):
        model = ComboConvBatchnormRelu6()
        self._test_conv_combo_tosa_MI_pipeline(model, model.get_inputs())

    def test_conv_batchnorm_relu6_tosa_BI(self):
        model = ComboConvBatchnormRelu6()
        self._test_conv_combo_tosa_BI_pipeline(model, model.get_inputs())

    @pytest.mark.corstone_fvp
    def test_conv_batchnorm_relu6_u55_BI(self):
        model = ComboConvBatchnormRelu6()
        self._test_conv_combo_ethos_BI_pipeline(
            model, common.get_u55_compile_spec(), model.get_inputs()
        )

    @pytest.mark.corstone_fvp
    def test_conv_batchnorm_relu_u85_BI(self):
        model = ComboConvBatchnormRelu6()
        self._test_conv_combo_ethos_BI_pipeline(
            model,
            common.get_u85_compile_spec(),
            model.get_inputs(),
        )

    ##################
    ## Conv + ReLU6 ##
    ##################
    @parameterized.expand(ComboConvRelu6.test_data)
    def test_conv_relu6_tosa_MI(self, test_data: torch.Tensor):
        model = ComboConvRelu6()
        test_data = (test_data,)
        self._test_conv_combo_tosa_MI_pipeline(model, test_data)

    @parameterized.expand(ComboConvRelu6.test_data)
    def test_conv_relu6_tosa_BI(self, test_data: torch.Tensor):
        model = ComboConvRelu6()
        test_data = (test_data,)
        self._test_conv_combo_tosa_BI_pipeline(model, test_data)

    @parameterized.expand(ComboConvRelu6.test_data)
    @pytest.mark.corstone_fvp
    def test_conv_relu6_u55_BI(self, test_data: torch.Tensor):
        model = ComboConvRelu6()
        test_data = (test_data,)
        self._test_conv_combo_ethos_BI_pipeline(
            model, common.get_u55_compile_spec(permute_memory_to_nhwc=True), test_data
        )

    @parameterized.expand(ComboConvRelu6.test_data)
    @pytest.mark.corstone_fvp
    def test_conv_relu6_u85_BI(self, test_data: torch.Tensor):
        model = ComboConvRelu6()
        test_data = (test_data,)
        self._test_conv_combo_ethos_BI_pipeline(
            model, common.get_u85_compile_spec(permute_memory_to_nhwc=True), test_data
        )

    ###############################
    ## Block bottleneck residual ##
    ###############################
    def test_block_bottleneck_residual_tosa_MI(self):
        model = ComboBlockBottleneckResidual()
        self._test_conv_combo_tosa_MI_pipeline(model, model.get_inputs())

    # TODO: Investigate flakyness (MLTORCH-307)
    @unittest.skip(reason="Skiped due to flakyness (MLTORCH-307)")
    def test_block_bottleneck_residual_tosa_BI(self):
        model = ComboBlockBottleneckResidual()
        self._test_conv_combo_tosa_BI_pipeline(model, model.get_inputs())

    @pytest.mark.corstone_fvp
    def test_block_bottleneck_residual_u55_BI(self):
        model = ComboBlockBottleneckResidual()
        self._test_conv_combo_ethos_BI_pipeline(
            model,
            common.get_u55_compile_spec(permute_memory_to_nhwc=True),
            model.get_inputs(),
        )

    @pytest.mark.corstone_fvp
    def test_block_bottleneck_residual_u85_BI(self):
        model = ComboBlockBottleneckResidual()
        self._test_conv_combo_ethos_BI_pipeline(
            model,
            common.get_u85_compile_spec(permute_memory_to_nhwc=True),
            model.get_inputs(),
        )

    ######################
    ## Conv + AvgPool2d ##
    ######################
    @parameterized.expand(ComboConvAvgPool2d.test_data)
    def test_conv_avgpool2d_tosa_MI(self, test_data: torch.Tensor):
        model = ComboConvAvgPool2d()
        test_data = (test_data,)
        self._test_conv_combo_tosa_MI_pipeline(model, test_data)

    @parameterized.expand(ComboConvAvgPool2d.test_data)
    def test_conv_avgpool2d_tosa_BI(self, test_data: torch.Tensor):
        model = ComboConvAvgPool2d()
        test_data = (test_data,)
        self._test_conv_combo_tosa_BI_pipeline(model, test_data)

    @parameterized.expand(ComboConvAvgPool2d.test_data)
    @pytest.mark.corstone_fvp
    def test_conv_avgpool2d_u55_BI(self, test_data: torch.Tensor):
        model = ComboConvAvgPool2d()
        test_data = (test_data,)
        self._test_conv_combo_ethos_BI_pipeline(
            model,
            common.get_u55_compile_spec(),
            test_data,
        )

    @parameterized.expand(ComboConvAvgPool2d.test_data)
    @pytest.mark.corstone_fvp
    def test_conv_avgpool2d_u85_BI(self, test_data: torch.Tensor):
        model = ComboConvAvgPool2d()
        test_data = (test_data,)
        self._test_conv_combo_ethos_BI_pipeline(
            model,
            common.get_u85_compile_spec(),
            test_data,
        )
