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
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
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
def test_minimum_tosa_MI(test_data: Tuple):
    TosaPipelineMI[test_t](Minimum(), test_data(), aten_op).run()


@common.parametrize("test_data", Minimum.test_parameters)
def test_minimum_tosa_BI(test_data: Tuple):
    TosaPipelineBI[test_t](Minimum(), test_data(), aten_op).run()


@common.parametrize("test_data", Minimum.test_parameters)
@common.XfailIfNoCorstone300
def test_minimum_u55_BI(test_data: Tuple):
    EthosU55PipelineBI[test_t](
        Minimum(),
        test_data(),
        aten_op,
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", Minimum.test_parameters)
@common.XfailIfNoCorstone320
def test_minimum_u85_BI(test_data: Tuple):
    EthosU85PipelineBI[test_t](
        Minimum(),
        test_data(),
        aten_op,
        run_on_fvp=True,
    ).run()
