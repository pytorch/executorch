# Copyright 2025 Arm Limited and/or its affiliates.
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

input_tt = Tuple[torch.Tensor, torch.Tensor]


def make_float_div_inputs(B: int = 4, T: int = 64) -> input_tt:
    x = torch.randn(B, T)
    # guard against zero in denominator
    y = torch.randn(B, T).abs() + 1e-3
    return x, y


class DivTensorModeFloat(torch.nn.Module):
    """
    torch.div(x, y, rounding_mode=mode) with
    mode from {None, "floor", "trunc"}.
    """

    aten_ops = ["aten.div.Tensor_mode"]
    aten_ops_int = ["aten.mul.Tensor", "aten.reciprocal.default"]

    def __init__(self, mode=None):
        super().__init__()
        assert mode in (None, "floor", "trunc")
        self.mode = mode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.div(x, y, rounding_mode=self.mode)


@pytest.mark.parametrize("mode", [None, "floor", "trunc"])
def test_div_tensor_mode_tosa_FP(mode):

    model = DivTensorModeFloat(mode)
    inputs = make_float_div_inputs()

    pipeline = TosaPipelineFP[input_tt](
        model,
        inputs,
        aten_op=model.aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@pytest.mark.parametrize("mode", [None, "floor", "trunc"])
def test_div_tensor_mode_tosa_INT(mode):

    model = DivTensorModeFloat(mode)
    inputs = make_float_div_inputs()

    pipeline = TosaPipelineINT[input_tt](
        model,
        inputs,
        aten_op=model.aten_ops_int,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()

@common.XfailIfNoCorstone300
@pytest.mark.parametrize("mode", [None, "floor"])
def test_div_tensor_mode_u55_INT(mode):

    model = DivTensorModeFloat(mode)
    inputs = make_float_div_inputs()

    pipeline = EthosU55PipelineINT[input_tt](
        model,
        inputs,
        aten_ops=model.aten_ops_int,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@pytest.mark.parametrize("mode", [None, "floor", "trunc"])
def test_div_tensor_mode_u85_INT(mode):

    model = DivTensorModeFloat(mode)
    inputs = make_float_div_inputs()

    pipeline = EthosU85PipelineINT[input_tt](
        model,
        inputs,
        aten_ops=model.aten_ops_int,
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
        run_on_fvp=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.parametrize("mode", [None, "floor", "trunc"])
def test_div_tensor_mode_vgf_INT(mode):

    model = DivTensorModeFloat(mode)
    inputs = make_float_div_inputs()

    pipeline = VgfPipeline[input_tt](
        model,
        inputs,
        aten_op=model.aten_ops_int,
        exir_op=[],
        tosa_version="TOSA-1.0+INT",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.parametrize("mode", [None, "floor", "trunc"])
def test_div_tensor_mode_vgf_FP(mode):

    model = DivTensorModeFloat(mode)
    inputs = make_float_div_inputs()

    pipeline = VgfPipeline[input_tt](
        model,
        inputs,
        aten_op=model.aten_ops,
        exir_op=[],
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()
