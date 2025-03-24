# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)
from parameterized import parameterized

test_t = tuple[torch.Tensor, torch.Tensor]


class MM(torch.nn.Module):
    test_data_generators = [
        lambda: (torch.rand(3, 5), torch.rand(5, 2)),
        lambda: (torch.rand(1, 1), torch.rand(1, 1)),
        lambda: (torch.ones(55, 3), torch.ones(3, 44)),
        lambda: (10000 * torch.randn(1, 10), torch.randn(10, 5)),
        lambda: (-10 * torch.randn(32, 64), 5 + 5 * torch.randn(64, 32)),
    ]
    aten_op = "torch.ops.aten.mm.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_mm_default"

    def forward(self, x, y):
        return torch.mm(x, y)


@parameterized.expand(MM.test_data_generators)
def test_mm_tosa_MI(test_data_generator: Callable[[], tuple]):
    test_data = test_data_generator()
    TosaPipelineMI[test_t](MM(), test_data, MM.aten_op).run()


@parameterized.expand(MM.test_data_generators)
def test_mm_tosa_BI(test_data_generator: Callable[[], tuple]):
    test_data = test_data_generator()
    TosaPipelineBI[test_t](MM(), test_data, MM.aten_op, MM.exir_op).run()


@parameterized.expand(MM.test_data_generators)
def test_mm_tosa_u55(test_data_generator: Callable[[], tuple]):
    test_data = test_data_generator()
    EthosU55PipelineBI[test_t](MM(), test_data, MM.aten_op).run()


@parameterized.expand(MM.test_data_generators)
def test_mm_tosa_u85(test_data_generator: Callable[[], tuple]):
    test_data = test_data_generator()
    EthosU85PipelineBI[test_t](MM(), test_data, MM.aten_op, MM.exir_op).run()


@parameterized.expand(MM.test_data_generators)
@common.SkipIfNoCorstone300
def test_mm_tosa_u55_on_fvp(test_data_generator: Callable[[], tuple]):
    test_data = test_data_generator()
    EthosU55PipelineBI[test_t](MM(), test_data, MM.aten_op, run_on_fvp=True).run()


@parameterized.expand(MM.test_data_generators)
@common.SkipIfNoCorstone320
def test_mm_tosa_u85_on_fvp(test_data_generator: Callable[[], tuple]):
    test_data = test_data_generator()
    EthosU85PipelineBI[test_t](
        MM(), test_data, MM.aten_op, MM.exir_op, run_on_fvp=True
    ).run()
