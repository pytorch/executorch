# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.arm._passes.meandim_to_averagepool_pass import (
    ConvertMeanDimToAveragePool,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from executorch.backends.xnnpack.test.tester.tester import RunPasses


class MeanDim(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=[-1, -2], keepdim=True)

    def get_inputs(self):
        return (torch.rand(1, 1280, 7, 7),)


class MeanDim2(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=1)

    def get_inputs(self):
        return (torch.rand(1, 1280, 7, 7),)


class TestMeandimToAveragePool2dPass(unittest.TestCase):
    """
    Tests the MeanDimToAveragePool2dPass which converts mean.dim to average_pool2d
    for the special case where dim is [-1, -2] and keepdim is True.
    """

    def test_tosa_BI_meandim_to_averagepool(self):
        module = MeanDim()
        test_pass_stage = RunPasses([ConvertMeanDimToAveragePool])
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge()
            .check(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .run_passes(test_pass_stage)
            .check(["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"])
        )

    def test_tosa_BI_meandim_no_modification(self):
        module = MeanDim2()
        test_pass_stage = RunPasses([ConvertMeanDimToAveragePool])
        (
            ArmTester(
                module,
                example_inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge()
            .check(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .run_passes(test_pass_stage)
            .check(["executorch_exir_dialects_edge__ops_aten_mean_dim"])
            .check_not(["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"])
        )
