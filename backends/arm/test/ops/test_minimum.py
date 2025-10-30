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
aten_op = "torch.ops.aten.minimum.default"


class Minimum(torch.nn.Module):
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
        return torch.minimum(x, y)


@common.parametrize("test_data", Minimum.test_parameters)
def test_minimum_tosa_FP(test_data: Tuple):
    TosaPipelineFP[test_t](Minimum(), test_data(), aten_op).run()


@common.parametrize("test_data", Minimum.test_parameters)
def test_minimum_tosa_INT(test_data: Tuple):
    TosaPipelineINT[test_t](Minimum(), test_data(), aten_op).run()


@common.parametrize("test_data", Minimum.test_parameters)
@common.XfailIfNoCorstone300
def test_minimum_u55_INT(test_data: Tuple):
    EthosU55PipelineINT[test_t](
        Minimum(),
        test_data(),
        aten_op,
    ).run()


@common.parametrize("test_data", Minimum.test_parameters)
@common.XfailIfNoCorstone320
def test_minimum_u85_INT(test_data: Tuple):
    EthosU85PipelineINT[test_t](
        Minimum(),
        test_data(),
        aten_op,
    ).run()


@common.parametrize("test_data", Minimum.test_parameters)
@common.SkipIfNoModelConverter
def test_minimum_vgf_FP(test_data: test_t):
    pipeline = VgfPipeline[test_t](Minimum(), test_data(), aten_op)
    pipeline.run()


@common.parametrize("test_data", Minimum.test_parameters)
@common.SkipIfNoModelConverter
def test_minimum_vgf_INT(test_data: test_t):
    pipeline = VgfPipeline[test_t](
        Minimum(),
        test_data(),
        aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
