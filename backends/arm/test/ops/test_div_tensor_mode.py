# Copyright 2025-2026 Arm Limited and/or its affiliates.
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

input_tt = Tuple[torch.Tensor, torch.Tensor]


class DivTensorModeFloat(torch.nn.Module):
    """torch.div(x, y, rounding_mode=mode) with mode in {None, "floor",
    "trunc"}.
    """

    aten_ops = ["aten.div.Tensor_mode"]
    aten_ops_int = ["aten.mul.Tensor", "aten.reciprocal.default"]

    def __init__(self, mode=None):
        super().__init__()
        assert mode in (None, "floor", "trunc")
        self.mode = mode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.div(x, y, rounding_mode=self.mode)


def _is_integer_rounded_div(mode, inputs) -> bool:
    if mode is None:
        return False
    for input in inputs:
        if isinstance(input, torch.Tensor):
            if input.dtype.is_floating_point:
                return False
        if not isinstance(input, int):
            return False
    return True


test_data = {
    "mode_none": lambda: (None, (torch.randn(4, 8), torch.randn(4, 8).abs() + 1e-3)),
    "mode_floor": lambda: (
        "floor",
        (torch.randn(4, 8), torch.randn(4, 8).abs() + 1e-3),
    ),
    "mode_trunc": lambda: (
        "trunc",
        (torch.randn(4, 8), torch.randn(4, 8).abs() + 1e-3),
    ),
    "int_denominator": lambda: (None, (torch.randn(4, 8), 2)),
    "int8_floor": lambda: (
        "floor",
        (
            (torch.randn(4, 8) * 100).to(dtype=torch.int8),
            (torch.rand(4, 8) * 100 + 10).to(dtype=torch.int8),
        ),
    ),
    "int8_int_scalar": lambda: (
        "floor",
        (
            (torch.randn(4, 8) * 100).to(dtype=torch.int8),
            9,
        ),
    ),
    "int8_float_scalar": lambda: (
        "floor",
        (
            (torch.randn(4, 8) * 100).to(dtype=torch.int8),
            9.5,
        ),
    ),
    "int16_trunc": lambda: (
        "trunc",
        (
            (torch.randn(4, 8) * 100).to(dtype=torch.int8),
            (torch.rand(4, 8) * 100 + 10).to(dtype=torch.int16),
        ),
    ),
    "int32_floor": lambda: (
        "floor",
        (
            (torch.randn(4, 8) * 100).to(dtype=torch.int32),
            (torch.rand(4, 8) * 100 + 10).to(dtype=torch.int32),
        ),
    ),
    "int32_trunc": lambda: (
        "trunc",
        (
            (torch.randn(4, 8) * 100).to(dtype=torch.int32),
            (torch.rand(4, 8) * 100 + 10).to(dtype=torch.int32),
        ),
    ),
}


@common.parametrize("data", test_data)
def test_div_tensor_mode_tosa_FP(data):
    mode, inputs = data()
    model = DivTensorModeFloat(mode)

    pipeline = TosaPipelineFP[input_tt](
        model,
        inputs,
        aten_op=model.aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.parametrize("data", test_data)
def test_div_tensor_mode_tosa_INT(data):
    mode, inputs = data()
    model = DivTensorModeFloat(mode)

    pipeline = TosaPipelineINT[input_tt](
        model,
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize(
    "data",
    test_data,
    xfails={
        "mode_trunc": "CPU op missing in unittests",
        "int16_trunc": "CPU op missing in unittests",
        "int32_trunc": "CPU op missing in unittests",
    },
)
def test_div_tensor_mode_u55_INT(data):
    mode, inputs = data()
    model = DivTensorModeFloat(mode)

    pipeline = EthosU55PipelineINT[input_tt](
        model,
        inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.tester.use_portable_ops = True
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("data", test_data)
def test_div_tensor_mode_u85_INT(data):
    mode, inputs = data()
    model = DivTensorModeFloat(mode)

    pipeline = EthosU85PipelineINT[input_tt](
        model,
        inputs,
        aten_ops=[],
        exir_ops=[],
        use_to_edge_transform_and_lower=True,
    )
    pipeline.tester.use_portable_ops = True
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("data", test_data)
def test_div_tensor_mode_vgf_quant(data):
    mode, inputs = data()
    model = DivTensorModeFloat(mode)

    pipeline = VgfPipeline[input_tt](
        model,
        inputs,
        aten_op=[],
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("data", test_data)
def test_div_tensor_mode_vgf_no_quant(data):
    mode, inputs = data()
    model = DivTensorModeFloat(mode)

    pipeline = VgfPipeline[input_tt](
        model,
        inputs,
        aten_op=model.aten_ops,
        exir_op=[],
        use_to_edge_transform_and_lower=True,
        quantize=False,
    )
    pipeline.run()
