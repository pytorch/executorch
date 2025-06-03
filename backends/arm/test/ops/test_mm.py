# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

test_t = tuple[torch.Tensor, torch.Tensor]


class MM(torch.nn.Module):
    test_data_generators = {
        "rand_2d": lambda: (torch.rand(3, 5), torch.rand(5, 2)),
        "rand_same": lambda: (torch.rand(1, 1), torch.rand(1, 1)),
        "ones": lambda: (torch.ones(55, 3), torch.ones(3, 44)),
        "randn_large": lambda: (10000 * torch.randn(1, 10), torch.randn(10, 5)),
        "rand_neg": lambda: (-10 * torch.randn(32, 64), 5 + 5 * torch.randn(64, 32)),
    }
    aten_op = "torch.ops.aten.mm.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_mm_default"

    def forward(self, x, y):
        return torch.mm(x, y)


@common.parametrize("test_data", MM.test_data_generators)
def test_mm_tosa_MI(test_data: Tuple):
    TosaPipelineMI[test_t](MM(), test_data(), MM.aten_op).run()


@common.parametrize("test_data", MM.test_data_generators)
def test_mm_tosa_BI(test_data: Tuple):
    TosaPipelineBI[test_t](MM(), test_data(), MM.aten_op, MM.exir_op, qtol=1).run()


@common.parametrize("test_data", MM.test_data_generators)
@common.XfailIfNoCorstone300
@pytest.mark.flaky  # Investigate flakiness (MLETORCH-870)
def test_mm_u55_BI(test_data: Tuple):
    EthosU55PipelineBI[test_t](
        MM(),
        test_data(),
        MM.aten_op,
        run_on_fvp=True,
    ).run()


@common.parametrize("test_data", MM.test_data_generators)
@common.XfailIfNoCorstone320
def test_mm_u85_BI(test_data: Tuple):
    EthosU85PipelineBI[test_t](
        MM(),
        test_data(),
        MM.aten_op,
        MM.exir_op,
        run_on_fvp=True,
    ).run()
