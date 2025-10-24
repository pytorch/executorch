# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor]  # Input x


class MeanDim(torch.nn.Module):
    """
    Basic mean model using torch.mean with keepdim = True
    """

    ops_after_pass = u55_ops_after_pass = {
        "torch.ops.aten.sum.dim_IntList": 2,
        "torch.ops.aten.mul.Tensor": 1,
    }

    ops_not_after_pass = u55_ops_not_after_pass = [
        "torch.ops.aten.reshape.default",
        "torch.ops.aten.avg_pool2d.default",
        "torch.ops.aten.mean.dim",
    ]

    def __init__(self):
        super(MeanDim, self).__init__()

    def forward(self, x):
        return torch.mean(x, (0, 1), True)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4, 4, 4),)


class MeanDimTensor(torch.nn.Module):
    """
    Basic mean model using torch.Tensor.mean with keepdim = False
    """

    ops_after_pass = {
        "torch.ops.aten.sum.dim_IntList": 2,
        "torch.ops.aten.mul.Tensor": 1,
        "torch.ops.aten.avg_pool2d.default": 1,
        "torch.ops.aten.reshape.default": 1,
    }

    ops_not_after_pass = [
        "torch.ops.aten.mean.dim",
    ]

    u55_ops_after_pass = {
        "torch.ops.aten.sum.dim_IntList": 2,
        "torch.ops.aten.mul.Tensor": 1,
        "torch.ops.aten.reshape.default": 1,
    }

    u55_ops_not_after_pass = [
        "torch.ops.aten.avg_pool2d.default",
        "torch.ops.aten.mean.dim",
    ]

    def __init__(self):
        super(MeanDimTensor, self).__init__()

    def forward(self, x):
        return x.mean((0, 2), False)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4, 4, 4),)


modules = {"meandim_basic": MeanDim(), "meandim_tensor": MeanDimTensor()}


@common.parametrize("module", modules)
def test_decompose_meandim_tosa_INT(module):
    # Decompose meandim_pass requires initiating the pas with args, which is not supported
    # by RunPasses in the arm_tester -> PassPipeline cannot be used.
    pipeline = TosaPipelineINT[input_t](
        module,
        module.get_inputs(),
        [],
    )
    pipeline.pop_stage("check_not.exir")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("to_executorch")
    pipeline.pop_stage("check.aten")
    pipeline.add_stage_after("quantize", pipeline.tester.check, module.ops_after_pass)
    pipeline.add_stage_after(
        "quantize", pipeline.tester.check_not, module.ops_not_after_pass
    )

    pipeline.dump_artifact("quantize")
    pipeline.run()


@common.parametrize("module", modules)
def test_decompose_meandim_u55_INT(module):
    # Decompose meandim_pass requires initiating the pas with args, which is not supported
    # by RunPasses in the arm_tester -> PassPipeline cannot be used.
    pipeline = EthosU55PipelineINT[input_t](
        module, module.get_inputs(), [], run_on_fvp=False
    )
    pipeline.pop_stage("check_not.exir")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("to_executorch")
    pipeline.pop_stage("check.aten")
    pipeline.add_stage_after(
        "quantize", pipeline.tester.check, module.u55_ops_after_pass
    )
    pipeline.add_stage_after(
        "quantize", pipeline.tester.check_not, module.u55_ops_not_after_pass
    )

    pipeline.dump_artifact("quantize")
    pipeline.run()
