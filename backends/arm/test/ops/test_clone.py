# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the clone op which copies the data of the input tensor (possibly with new data format)
#

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

aten_op = "torch.ops.aten.clone.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_clone_default"

input_t = Tuple[torch.Tensor]


class Clone(torch.nn.Module):
    """A simple module that clones an input tensor."""

    def forward(self, x: torch.Tensor):
        return x.clone()


test_data_suite = {
    "ones_1D_10": lambda: (torch.ones(10),),
    "ones_1D_50": lambda: (torch.ones(50),),
    "rand_1D_20": lambda: (torch.rand(20),),
    "rand_2D_10x10": lambda: (torch.rand(10, 10),),
    "rand_3D_5x5x5": lambda: (torch.rand(5, 5, 5),),
    "rand_4D_2x3x4x5": lambda: (torch.rand(2, 3, 4, 5),),
    "large_tensor": lambda: (torch.rand(1000),),
}


@common.parametrize("test_data", test_data_suite)
def test_clone_tosa_MI(test_data: Tuple[torch.Tensor]):

    pipeline = TosaPipelineMI[input_t](
        Clone(),
        test_data(),
        aten_op,
        exir_op,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_clone_tosa_BI(test_data):
    pipeline = TosaPipelineBI[input_t](
        Clone(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="Empty subgraph leads to Vela compilation failure. See: https://jira.arm.com/browse/MLBEDSW-10477"
)
def test_clone_u55_BI(test_data):
    pipeline = EthosU55PipelineBI[input_t](
        Clone(),
        test_data(),
        aten_op,
        exir_op,
        run_on_fvp=True,
    )

    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason="Empty subgraph leads to Vela compilation failure. See: https://jira.arm.com/browse/MLBEDSW-10477"
)
def test_clone_u85_BI(test_data):
    pipeline = EthosU85PipelineBI[input_t](
        Clone(),
        test_data(),
        aten_op,
        exir_op,
        run_on_fvp=True,
    )

    pipeline.run()
