# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_meandim_pass import DecomposeMeanDimPass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class MeanDim(torch.nn.Module):
    """
    Basic mean model using torch.mean function making sure keepdim=True (keepdim=False doesnt work for this pass for some reason)
    """

    def __init__(self):
        super(MeanDim, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, True)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


class MeanDimTensor(torch.nn.Module):
    """
    Basic mean model using torch.Tensor.mean function making sure keepdim=True (keepdim=False doesnt work for this pass for some reason)
    """

    def __init__(self):
        super(MeanDimTensor, self).__init__()

    def forward(self, x):
        return x.mean(1, True)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


modules = {"meandim_basic": MeanDim(), "meandim_tensor": MeanDimTensor()}


@common.parametrize("module", modules)
def test_decompose_meandim_tosa_MI(module):
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        tosa_version="TOSA-0.80+MI",
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
        },
        ops_not_before_pass=[
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
            "executorch_exir_dialects_edge__ops_aten_full_default",
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList",
        ],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_full_default": 1,
            "executorch_exir_dialects_edge__ops_aten_sum_dim_IntList": 1,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_aten_mean_dim"],
        pass_list=[DecomposeMeanDimPass],
    )
    pipeline.run()
