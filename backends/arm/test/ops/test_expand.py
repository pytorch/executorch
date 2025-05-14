# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the expand op which copies the data of the input tensor (possibly with new data format)
#


from typing import Sequence, Tuple

import pytest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)

aten_op = "torch.ops.aten.expand.default"
input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x, Input y


class Expand(torch.nn.Module):
    # (input tensor, multiples)
    test_parameters = {
        "rand_1d_both": lambda: (torch.rand(1), (2,)),
        "rand_1d": lambda: (torch.randn(1), (2, 2, 4)),
        "rand_4d": lambda: (torch.randn(1, 1, 1, 5), (1, 4, -1, -1)),
        "rand_batch_1": lambda: (torch.randn(1, 1), (1, 2, 2, 4)),
        "rand_batch_2": lambda: (torch.randn(1, 1), (2, 2, 2, 4)),
        "rand_mix_neg": lambda: (torch.randn(10, 1, 1, 97), (-1, 4, -1, -1)),
        "rand_small_neg": lambda: (torch.rand(1, 1, 2, 2), (4, 3, -1, 2)),
    }

    test_reject_set = {
        "rand_2d": lambda: (torch.randn(1, 4), (1, -1)),
        "rand_neg_mul": lambda: (torch.randn(1, 1, 192), (1, -1, -1)),
    }

    def forward(self, x: torch.Tensor, m: Sequence):
        return x.expand(m)


@common.parametrize("test_data", Expand.test_parameters | Expand.test_reject_set)
def test_expand_tosa_MI(test_data: Tuple):
    pipeline = TosaPipelineMI[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_parameters | Expand.test_reject_set)
def test_expand_tosa_BI(test_data: Tuple):
    pipeline = TosaPipelineBI[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


x_fails = {
    "rand_batch_2": "AssertionError: Output 0 does not match reference output.",
    "rand_mix_neg": "AssertionError: Output 0 does not match reference output.",
    "rand_small_neg": "AssertionError: Output 0 does not match reference output.",
}


@common.parametrize("test_data", Expand.test_parameters, x_fails)
@common.XfailIfNoCorstone300
def test_expand_u55_BI(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_parameters, x_fails)
@common.XfailIfNoCorstone320
def test_expand_u85_BI(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_reject_set)
@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="MLETORCH-716: Node will be optimized away and Vela can't handle empty graphs"
)
def test_expand_u55_BI_failure_set(test_data: Tuple):
    pipeline = EthosU55PipelineBI[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Expand.test_reject_set)
@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason="MLETORCH-716: Node will be optimized away and Vela can't handle empty graphs"
)
def test_expand_u85_BI_failure_set(test_data: Tuple):
    pipeline = EthosU85PipelineBI[input_t1](
        Expand(),
        test_data(),
        aten_op,
        exir_ops=[],
        run_on_fvp=True,
    )
    pipeline.run()
