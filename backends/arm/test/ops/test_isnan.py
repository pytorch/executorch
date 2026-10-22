# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
    VgfPipeline,
)

aten_op = "torch.ops.aten.isnan.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_isnan_default"


class IsNan(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.isnan(x)


test_data_suite = {
    "finite": lambda: torch.tensor([-1.0, 0.0, 3.14]),
    "nan": lambda: torch.tensor([float("nan"), 0.0, float("inf")]),
    "integer": lambda: torch.tensor([-5, 0, 9], dtype=torch.int32),
    "rank4": lambda: torch.tensor([[[[float("nan"), 0.0]]], [[[float("inf"), -3.0]]]]),
}


@common.parametrize(
    "test_data",
    {name: data for name, data in test_data_suite.items() if name != "integer"},
)
def test_isnan_tosa_FP(test_data: Callable[[], torch.Tensor]) -> None:
    TosaPipelineFP(
        IsNan(),
        (test_data(),),
        aten_op,
        exir_op,
    ).run()


def test_isnan_tosa_FP_falls_back_for_integer() -> None:
    OpNotSupportedPipeline(
        IsNan(),
        (test_data_suite["integer"](),),
        {exir_op: 1},
        quantize=False,
    ).run()


def test_isnan_tosa_INT_falls_back() -> None:
    test_data = (test_data_suite["nan"](),)
    pipeline = OpNotSupportedPipeline(
        IsNan(),
        test_data,
        {exir_op: 1},
        quantize=True,
    )
    quantize_stage = pipeline._stages[pipeline.find_pos("quantize")].args[0]
    quantize_stage.calibration_samples = [(torch.ones_like(test_data[0]),)]
    pipeline.run()


@common.SkipIfNoModelConverter
def test_isnan_vgf_no_quant() -> None:
    VgfPipeline(
        IsNan(),
        (test_data_suite["nan"](),),
        aten_op,
        exir_op,
        quantize=False,
    ).run()
