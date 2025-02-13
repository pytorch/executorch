# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm._passes.meandim_to_averagepool_pass import (
    ConvertMeanDimToAveragePoolPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TestPassPipeline


input_t = Tuple[torch.Tensor, torch.Tensor]  # Input x


class MeanDim(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=[-1, -2], keepdim=True)

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 1280, 7, 7),)

    ops_before_pass = {"executorch_exir_dialects_edge__ops_aten_mean_dim": 1}
    ops_after_pass = {"executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1}
    ops_not_after_pass = [
        "aten_sum_dim_int_list",
        "aten_full_default",
        "aten_mul_tensor",
    ]


class MeanDim2(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=1)

    def get_inputs(self) -> input_t:
        return (torch.rand(1, 1280, 7, 7),)

    ops_before_pass = {
        "aten_sum_dim_int_list": 3,
        "aten_full_default": 4,
        "aten_mul_tensor": 3,
    }
    ops_after_pass = {
        "aten_sum_dim_int_list": 3,
        "aten_full_default": 4,
        "aten_mul_tensor": 3,
    }
    ops_not_after_pass = ["executorch_exir_dialects_edge__ops_aten_avg_pool2d_default"]


modules = {
    "meandim_to_averagepool": MeanDim(),
    "meandim_no_modification": MeanDim2(),
}


@common.parametrize("module", modules)
def test_meandim_to_avgpool_tosa_BI(module):
    """
    Tests the MeanDimToAveragePool2dPass which converts mean.dim to average_pool2d
    for the special case where dim is [-1, -2] and keepdim is True.
    """
    pipeline = TestPassPipeline[input_t](
        module,
        module.get_inputs(),
        tosa_version="TOSA-0.80+BI",
        ops_before_pass=module.ops_before_pass,
        ops_after_pass=module.ops_after_pass,
        ops_not_after_pass=module.ops_not_after_pass,
        pass_list=[ConvertMeanDimToAveragePoolPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output
    pipeline.run()
