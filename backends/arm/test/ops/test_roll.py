# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


input_t1 = Tuple[torch.Tensor]
aten_op = "torch.ops.aten.roll.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_roll_default"

test_data_fp = {
    "bev_cyclic_shift_fp32": lambda: (
        torch.randn(1, 8, 8, 4, dtype=torch.float32),
        (-2, -2),
        (1, 2),
    ),
    "bev_cyclic_shift_fp16": lambda: (
        torch.randn(1, 8, 8, 4, dtype=torch.float16),
        (-2, -2),
        (1, 2),
    ),
}

test_data_bf16 = {
    "bev_cyclic_shift_bf16": lambda: (
        torch.randn(1, 8, 8, 4, dtype=torch.bfloat16),
        (-2, -2),
        (1, 2),
    ),
}

test_data_quant = {
    "bev_cyclic_shift": lambda: (
        torch.randn(1, 8, 8, 4, dtype=torch.float32),
        (-2, -2),
        (1, 2),
    ),
}


class Roll(torch.nn.Module):
    def __init__(self, shifts: tuple[int, ...], dims: tuple[int, ...]) -> None:
        super().__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, self.shifts, self.dims)


@common.parametrize("test_data", test_data_fp)
def test_roll_tosa_FP(test_data) -> None:
    data, shifts, dims = test_data()
    pipeline = TosaPipelineFP[input_t1](Roll(shifts, dims), (data,), aten_op, exir_op)
    pipeline.count_tosa_ops({"SLICE": 4, "CONCAT": 2})
    pipeline.run()


@common.parametrize("test_data", test_data_bf16)
def test_roll_tosa_FP_bf16(test_data) -> None:
    data, shifts, dims = test_data()
    pipeline = TosaPipelineFP[input_t1](
        Roll(shifts, dims),
        (data,),
        aten_op,
        exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.count_tosa_ops({"SLICE": 4, "CONCAT": 2})
    pipeline.run()


@common.parametrize("test_data", test_data_quant)
def test_roll_tosa_INT(test_data) -> None:
    data, shifts, dims = test_data()
    pipeline = TosaPipelineINT[input_t1](Roll(shifts, dims), (data,), aten_op, exir_op)
    pipeline.count_tosa_ops({"SLICE": 4, "CONCAT": 2})
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_bf16)
@common.SkipIfNoModelConverter
def test_roll_vgf_no_quant(test_data) -> None:
    data, shifts, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        Roll(shifts, dims),
        (data,),
        aten_op,
        exir_op,
        quantize=False,
        run_on_vulkan_runtime=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_quant)
@common.SkipIfNoModelConverter
def test_roll_vgf_quant(test_data) -> None:
    data, shifts, dims = test_data()
    pipeline = VgfPipeline[input_t1](
        Roll(shifts, dims),
        (data,),
        aten_op,
        exir_op,
        quantize=True,
        run_on_vulkan_runtime=False,
    )
    pipeline.run()
