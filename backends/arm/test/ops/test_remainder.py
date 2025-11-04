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


def _nonzero_float_tensor(*shape: int) -> torch.Tensor:
    return torch.rand(*shape, dtype=torch.float32) * 5 + 0.1


class Remainder(torch.nn.Module):
    input_t = Tuple[torch.Tensor | float, torch.Tensor | float]

    test_cases = {
        "rank2_tensors": lambda: (
            torch.randn(2, 3) * 7,
            _nonzero_float_tensor(2, 3),
        ),
        "rank4_tensors": lambda: (
            torch.randn(1, 4, 2, 3) * 7,
            _nonzero_float_tensor(1, 4, 2, 3),
        ),
        "broadcast": lambda: (
            torch.randn(4, 5, 1),
            _nonzero_float_tensor(1, 5, 6),
        ),
        "scalar_rhs": lambda: (
            torch.randn(1, 2, 3, 4),
            0.25,
        ),
    }

    def forward(self, x: torch.Tensor | float, y: torch.Tensor | float) -> torch.Tensor:
        return torch.remainder(x, y)


def _get_aten_op(test_data: Remainder.input_t):
    if any(isinstance(x, float) for x in test_data):
        return "torch.ops.aten.remainder.Scalar"
    else:
        return "torch.ops.aten.remainder.Tensor"


def _get_exir_op(test_data: Remainder.input_t):
    if isinstance(test_data[1], float):
        return "executorch_exir_dialects_edge__ops_aten_remainder_Scalar"
    else:
        return "executorch_exir_dialects_edge__ops_aten_remainder_Tensor"


@common.parametrize("test_data", Remainder.test_cases)
def test_remainder_tosa_FP(test_data):
    data = test_data()
    pipeline = TosaPipelineFP[Remainder.input_t](
        Remainder(),
        data,
        _get_aten_op(data),
        _get_exir_op(data),
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases)
def test_remainder_tosa_INT(test_data):
    pipeline = TosaPipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases)
@common.XfailIfNoCorstone300
def test_remainder_u55_INT(test_data):
    pipeline = EthosU55PipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases)
@common.XfailIfNoCorstone320
def test_remainder_u85_INT(test_data):
    pipeline = EthosU85PipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases)
@common.SkipIfNoModelConverter
def test_remainder_vgf_FP(test_data):
    data = test_data()
    pipeline = VgfPipeline[Remainder.input_t](
        Remainder(),
        data,
        _get_aten_op(data),
        _get_exir_op(data),
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases)
@common.SkipIfNoModelConverter
def test_remainder_vgf_INT(test_data):
    pipeline = VgfPipeline[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
