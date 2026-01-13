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

input_t1 = Tuple[torch.Tensor]


class AliasCopy(torch.nn.Module):
    """
    Tests proper handling of alias_copy when used directly.

    alias_copy can also appear from PyTorch/ExecuTorch optimizations
    such as `x.transpose(0, 0)`. This is optimized to an alias_copy but
    not before dq/q operators are added.
    """

    aten_op = "torch.ops.aten.alias_copy.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_alias_copy_default"

    test_data: dict[input_t1] = {
        "1d_ramp": lambda: (torch.arange(-16, 16, 0.2),),
        "2d_ones": lambda: (torch.ones(5, 5),),
        "3d_rand": lambda: (torch.rand(3, 5, 5),),
        "4d_zeros": lambda: (torch.zeros(1, 10, 10, 10),),
    }

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return (
            torch.alias_copy(x) * 1
        )  # Multiply by one to make sure it is partitioned.


@common.parametrize("test_data", AliasCopy.test_data)
def test_alias_tosa_FP(test_data: input_t1):
    TosaPipelineFP[input_t1](
        AliasCopy(),
        test_data(),
        AliasCopy.aten_op,
        AliasCopy.exir_op,
    ).run()


@common.parametrize("test_data", AliasCopy.test_data)
def test_alias_tosa_INT(test_data: input_t1):
    TosaPipelineINT[input_t1](
        AliasCopy(),
        test_data(),
        AliasCopy.aten_op,
        AliasCopy.exir_op,
    ).run()


@common.parametrize("test_data", AliasCopy.test_data)
@common.XfailIfNoCorstone300
def test_alias_u55_INT(test_data: input_t1):
    EthosU55PipelineINT[input_t1](
        AliasCopy(),
        test_data(),
        AliasCopy.aten_op,
        AliasCopy.exir_op,
    ).run()


@common.parametrize("test_data", AliasCopy.test_data)
@common.XfailIfNoCorstone320
def test_alias_u85_INT(test_data: input_t1):
    EthosU85PipelineINT[input_t1](
        AliasCopy(),
        test_data(),
        AliasCopy.aten_op,
        AliasCopy.exir_op,
    ).run()


@common.parametrize("test_data", AliasCopy.test_data)
@common.SkipIfNoModelConverter
def test_alias_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AliasCopy(),
        test_data(),
        AliasCopy.aten_op,
        AliasCopy.exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", AliasCopy.test_data)
@common.SkipIfNoModelConverter
def test_alias_vgf_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        AliasCopy(),
        test_data(),
        AliasCopy.aten_op,
        AliasCopy.exir_op,
        quantize=True,
    )
    pipeline.run()
