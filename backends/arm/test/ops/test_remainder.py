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

    aten_op_tensor = "torch.ops.aten.remainder.Tensor"
    exir_op_tensor = "executorch_exir_dialects_edge__ops_aten_remainder_Tensor"
    aten_op_scalar = "torch.ops.aten.remainder.Scalar"
    exir_op_scalar = "executorch_exir_dialects_edge__ops_aten_remainder_Scalar"

    test_cases_tensor = {
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
    }

    test_cases_scalar = {
        "scalar_pos": lambda: (
            torch.randn(1, 2, 3, 4),
            0.25,
        ),
        "scalar_neg": lambda: (
            torch.randn(3, 4),
            -0.25,
        ),
    }

    def forward(self, x: torch.Tensor | float, y: torch.Tensor | float) -> torch.Tensor:
        return torch.remainder(x, y)


@common.parametrize("test_data", Remainder.test_cases_tensor)
def test_remainder_tensor_tosa_FP(test_data):
    data = test_data()
    pipeline = TosaPipelineFP[Remainder.input_t](
        Remainder(),
        data,
        Remainder.aten_op_tensor,
        Remainder.exir_op_tensor,
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_scalar)
def test_remainder_scalar_tosa_FP(test_data):
    data = test_data()
    pipeline = TosaPipelineFP[Remainder.input_t](
        Remainder(),
        data,
        Remainder.aten_op_scalar,
        Remainder.exir_op_scalar,
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_tensor)
def test_remainder_tensor_tosa_INT(test_data):
    pipeline = TosaPipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_scalar)
def test_remainder_scalar_tosa_INT(test_data):
    pipeline = TosaPipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_tensor)
@common.XfailIfNoCorstone300
def test_remainder_tensor_u55_INT(test_data):
    pipeline = EthosU55PipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_scalar)
@common.XfailIfNoCorstone300
def test_remainder_scalar_u55_INT(test_data):
    pipeline = EthosU55PipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_tensor)
@common.XfailIfNoCorstone320
def test_remainder_tensor_u85_INT(test_data):
    pipeline = EthosU85PipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_scalar)
@common.XfailIfNoCorstone320
def test_remainder_scalar_u85_INT(test_data):
    pipeline = EthosU85PipelineINT[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_tensor)
@common.SkipIfNoModelConverter
def test_remainder_tensor_vgf_FP(test_data):
    data = test_data()
    pipeline = VgfPipeline[Remainder.input_t](
        Remainder(),
        data,
        Remainder.aten_op_tensor,
        Remainder.exir_op_tensor,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_scalar)
@common.SkipIfNoModelConverter
def test_remainder_scalar_vgf_FP(test_data):
    data = test_data()
    pipeline = VgfPipeline[Remainder.input_t](
        Remainder(),
        data,
        Remainder.aten_op_scalar,
        Remainder.exir_op_scalar,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_tensor)
@common.SkipIfNoModelConverter
def test_remainder_tensor_vgf_INT(test_data):
    pipeline = VgfPipeline[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@common.parametrize("test_data", Remainder.test_cases_scalar)
@common.SkipIfNoModelConverter
def test_remainder_scalar_vgf_INT(test_data):
    pipeline = VgfPipeline[Remainder.input_t](
        Remainder(),
        test_data(),
        [],
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
