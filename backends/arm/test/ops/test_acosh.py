# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import pytest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t = Tuple[torch.Tensor]  # Input x
aten_op = "torch.ops.aten.acosh.default"


test_data_suite = {
    # Valid input cases
    "ones": lambda: torch.ones(1, 7, 10, 12),
    "just_above_one": lambda: torch.tensor([1.0001, 1.01, 1.1, 2.0]),
    "rand_valid": lambda: torch.rand(10, 10) * 10 + 1,  # [1, 11)
    "ramp_valid": lambda: torch.linspace(1.0, 20.0, steps=160),
    "large": lambda: torch.tensor([10.0, 100.0, 1000.0, 1e6]),
    "mixed_valid": lambda: torch.tensor([1.0, 2.0, 10.0, 100.0]),
}

test_data_suite_xfails = {
    # Invalid input cases (should return nan or error)
    "zeros": lambda: torch.zeros(1, 5, 3, 2),
    "neg_ones": lambda: -torch.ones(10, 10, 10),
    "rand_invalid": lambda: torch.rand(10, 10),  # [0, 1)
    "ramp_invalid": lambda: torch.linspace(-10.0, 0.99, steps=160),
    "near_zero": lambda: torch.tensor([-1e-6, 0.0, 1e-6]),
    "large_negative": lambda: torch.tensor([-100.0, -10.0, 0.0]),
}


class Acosh(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return torch.acosh(x)


@common.parametrize("test_data", test_data_suite)
def test_acosh_tosa_FP(test_data: Tuple):
    pipeline = TosaPipelineFP[input_t](
        Acosh(),
        (test_data(),),
        aten_op,
        exir_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_acosh_tosa_INT(test_data: Tuple):
    pipeline = TosaPipelineINT[input_t](
        Acosh(),
        (test_data(),),
        aten_op=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_acosh_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t](
        Acosh(),
        (test_data(),),
        aten_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_xfails)
@pytest.mark.xfail(reason="Invalid inputs are currently not handled")
def test_acosh_u55_INT_xfail(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t](
        Acosh(),
        (test_data(),),
        aten_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_acosh_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t](
        Acosh(),
        (test_data(),),
        aten_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite_xfails)
@pytest.mark.xfail(reason="Invalid inputs are currently not handled")
def test_acosh_u85_INT_xfail(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t](
        Acosh(),
        (test_data(),),
        aten_ops=[],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_acosh_vgf_FP(test_data: Tuple):
    pipeline = VgfPipeline[input_t](
        Acosh(),
        (test_data(),),
        aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_acosh_vgf_INT(test_data: Tuple):
    pipeline = VgfPipeline[input_t](
        Acosh(),
        (test_data(),),
        aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
