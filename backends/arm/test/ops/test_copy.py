# Copyright 2025 Arm Limited and/or its affiliates.
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


class CopyOutput(torch.nn.Module):
    def forward(self, x):
        y = torch.zeros(x.shape)
        return y.copy_(x / x) + x


class CopyFirstArg(torch.nn.Module):
    def forward(self, x):
        y = torch.zeros(x.shape)
        return y.copy_(x) + x


class CopySecondArg(torch.nn.Module):
    def forward(self, x):
        y = torch.zeros(x.shape)
        return x * y.copy_(x)


class CopyBothArgs(torch.nn.Module):
    def forward(self, x):
        y = torch.zeros(x.shape)
        return y.copy_(x) + y.copy_(x)


class CopyAfterOtherOp(torch.nn.Module):
    def forward(self, x):
        y = torch.zeros(x.shape)
        x = x * 2
        return y.copy_(x) + x


class CopyParallelToOtherOp(torch.nn.Module):
    def forward(self, x):
        y = torch.zeros(x.shape)
        return x * 2 + y.copy_(x)


test_suite = {
    "copy_output": lambda: (
        CopyOutput,
        (torch.rand(1, 2, 3, 4, dtype=torch.float32),),
    ),
    "copy_first_arg": lambda: (
        CopyFirstArg,
        (torch.rand(1, 2, 3, 4, dtype=torch.float32),),
    ),
    "copy_second_arg": lambda: (
        CopySecondArg,
        (torch.rand(1, 2, 3, 4, dtype=torch.float32),),
    ),
    "copy_both_args": lambda: (
        CopyBothArgs,
        (torch.rand(1, 2, 3, 4, dtype=torch.float32),),
    ),
    "copy_after_other_op": lambda: (
        CopyAfterOtherOp,
        (torch.rand(1, 2, 3, 4, dtype=torch.float32),),
    ),
    "copy_parallel_to_other_op": lambda: (
        CopyParallelToOtherOp,
        (torch.rand(1, 2, 3, 4, dtype=torch.float32),),
    ),
}


aten_op = "torch.ops.aten.copy_.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_copy_default"

input_t = Tuple[torch.Tensor]


@common.parametrize("input_data", test_suite)
def test_copy_tosa_FP(input_data):
    module, input_tensor = input_data()
    pipeline = TosaPipelineFP[input_t](
        module(),
        input_tensor,
        aten_op=aten_op,
        exir_op=exir_op,
    )
    pipeline.run()


@common.parametrize("input_data", test_suite)
def test_copy_tosa_INT(input_data):
    module, input_tensor = input_data()

    pipeline = TosaPipelineINT[input_t](
        module(),
        input_tensor,
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("input_data", test_suite)
@common.XfailIfNoCorstone300
def test_copy_u55_INT(input_data):
    module, input_tensor = input_data()

    pipeline = EthosU55PipelineINT[input_t](
        module(),
        input_tensor,
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("input_data", test_suite)
@common.XfailIfNoCorstone320
def test_copy_u85_INT(input_data):
    module, input_tensor = input_data()

    pipeline = EthosU85PipelineINT[input_t](
        module(),
        input_tensor,
        aten_op,
        exir_op,
    )

    pipeline.run()


@common.parametrize("test_data", test_suite)
@common.SkipIfNoModelConverter
def test_copy_vgf_no_quant(test_data):
    module, input_tensor = test_data()
    pipeline = VgfPipeline[input_t](
        module(),
        input_tensor,
        aten_op=aten_op,
        exir_op=exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_suite)
@common.SkipIfNoModelConverter
def test_copy_vgf_quant(test_data):
    module, input_tensor = test_data()
    pipeline = VgfPipeline[input_t](
        module(),
        input_tensor,
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
