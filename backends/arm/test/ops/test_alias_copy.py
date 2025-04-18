# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
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
        "1d_ramp": (torch.arange(-16, 16, 0.2),),
        "2d_ones": (torch.ones(5, 5),),
        "3d_rand": (torch.rand(3, 5, 5),),
        "4d_zeros": (torch.zeros(1, 10, 10, 10),),
    }

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.alias_copy(x)


@common.parametrize("test_data", AliasCopy.test_data)
def test_alias_copy_tosa_MI(test_data: input_t1):
    TosaPipelineMI[input_t1](
        AliasCopy(),
        test_data,
        AliasCopy.aten_op,
        AliasCopy.exir_op,
    ).run()


@common.parametrize("test_data", AliasCopy.test_data)
def test_alias_copy_tosa_BI(test_data: input_t1):
    TosaPipelineBI[input_t1](
        AliasCopy(),
        test_data,
        AliasCopy.aten_op,
        AliasCopy.exir_op,
    ).run()


@common.parametrize("test_data", AliasCopy.test_data)
def test_alias_copy_u55_BI(test_data: input_t1):
    EthosU55PipelineBI[input_t1](
        AliasCopy(),
        test_data,
        AliasCopy.aten_op,
        AliasCopy.exir_op,
    ).run()


@common.parametrize("test_data", AliasCopy.test_data)
def test_alias_copy_u85_BI(test_data: input_t1):
    EthosU85PipelineBI[input_t1](
        AliasCopy(),
        test_data,
        AliasCopy.aten_op,
        AliasCopy.exir_op,
    ).run()
