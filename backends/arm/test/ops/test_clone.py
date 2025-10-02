# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.clone.default"
exir_op = "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default"

input_t = Tuple[torch.Tensor]


class CloneFirstArg(torch.nn.Module):
    def forward(self, x):
        return x.clone() + x


class CloneSecondArg(torch.nn.Module):
    def forward(self, x):
        return x * x.clone()


class CloneOutput(torch.nn.Module):
    def forward(self, x):
        return (x / x).clone()


class CloneBothArgs(torch.nn.Module):
    def forward(self, x):
        return x.clone() + x.clone()


class CloneAfterOtherOp(torch.nn.Module):
    def forward(self, x):
        x = x * 2
        return x.clone() + x


class CloneParallelToOtherOp(torch.nn.Module):
    def forward(self, x):
        return x * 2 + x.clone()


delegated_clones = {
    "clone_first_arg": lambda: (CloneFirstArg, (torch.rand(1, 2, 3, 4),)),
    "clone_second_arg": lambda: (CloneSecondArg, (torch.rand(1, 2, 3, 4),)),
    "clone_output": lambda: (CloneOutput, (torch.rand(1, 2, 3, 4),)),
    "clone_both_args": lambda: (CloneBothArgs, (torch.rand(1, 2, 3, 4),)),
    "clone_after_other_op": lambda: (CloneAfterOtherOp, (torch.rand(1, 2, 3, 4),)),
    "clone_parallel_to_other_op": lambda: (
        CloneParallelToOtherOp,
        (torch.rand(1, 2, 3, 4),),
    ),
}


@common.parametrize("input_data", delegated_clones)
def test_clone_tosa_FP(input_data):
    module, input_tensor = input_data()
    pipeline = TosaPipelineFP[input_t](
        module(),
        input_tensor,
        [],
    )
    pipeline.run()


@common.parametrize("input_data", delegated_clones)
def test_clone_tosa_INT(input_data):
    module, input_tensor = input_data()

    pipeline = TosaPipelineINT[input_t](
        module(),
        input_tensor,
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("input_data", delegated_clones)
@common.XfailIfNoCorstone300
def test_clone_u55_INT(input_data):
    module, input_tensor = input_data()

    pipeline = EthosU55PipelineINT[input_t](
        module(),
        input_tensor,
        aten_op,
        exir_op,
    )

    pipeline.run()


@common.parametrize("input_data", delegated_clones)
@common.XfailIfNoCorstone320
def test_clone_u85_INT(input_data):
    module, input_tensor = input_data()

    pipeline = EthosU85PipelineINT[input_t](
        module(),
        input_tensor,
        aten_op,
        exir_op,
    )

    pipeline.run()


@common.parametrize("test_data", delegated_clones)
@common.SkipIfNoModelConverter
def test_clone_vgf_FP(test_data):
    module, input_tensor = test_data()
    pipeline = VgfPipeline[input_t](
        module(), input_tensor, aten_op, exir_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_data", delegated_clones)
@common.SkipIfNoModelConverter
def test_clone_vgf_INT(test_data):
    module, input_tensor = test_data()
    pipeline = VgfPipeline[input_t](
        module(),
        input_tensor,
        aten_op,
        exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
