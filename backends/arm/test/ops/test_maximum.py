# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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

test_t = tuple[torch.Tensor, torch.Tensor]
aten_op = "torch.ops.aten.maximum.default"


class Maximum(torch.nn.Module):
    test_parameters = {
        "float_tensor": lambda: (
            torch.FloatTensor([1, 2, 3, 5, 7]),
            (torch.FloatTensor([2, 1, 2, 1, 10])),
        ),
        "ones": lambda: (torch.ones(1, 10, 4, 6), 2 * torch.ones(1, 10, 4, 6)),
        "rand_diff": lambda: (torch.randn(1, 1, 4, 4), torch.ones(1, 1, 4, 1)),
        "rand_same": lambda: (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)),
        "rand_large": lambda: (
            10000 * torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 1),
        ),
    }

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.maximum(x, y)


@common.parametrize("test_data", Maximum.test_parameters)
def test_maximum_tosa_FP(test_data: Tuple):
    TosaPipelineFP[test_t](Maximum(), test_data(), aten_op).run()


@common.parametrize("test_data", Maximum.test_parameters)
def test_maximum_tosa_INT(test_data: Tuple):
    TosaPipelineINT[test_t](Maximum(), test_data(), aten_op).run()


@common.parametrize("test_data", Maximum.test_parameters)
@common.XfailIfNoCorstone300
def test_maximum_u55_INT(test_data: Tuple):
    EthosU55PipelineINT[test_t](
        Maximum(),
        test_data(),
        aten_op,
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", Maximum.test_parameters)
@common.XfailIfNoCorstone320
def test_maximum_u85_INT(test_data: Tuple):
    EthosU85PipelineINT[test_t](
        Maximum(),
        test_data(),
        aten_op,
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", Maximum.test_parameters)
@common.SkipIfNoModelConverter
def test_maximum_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[test_t](
        Maximum(),
        test_data(),
        aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Maximum.test_parameters)
@common.SkipIfNoModelConverter
def test_maximum_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[test_t](
        Maximum(),
        test_data(),
        aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
