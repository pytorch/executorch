# Copyright 2025 Arm Limited and/or its affiliates.
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

aten_op = "torch.ops.aten.log1p.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_log1p_default"

input_t1 = Tuple[torch.Tensor]


def _tensor(values):
    return torch.tensor(values, dtype=torch.float32)


test_data_suite = {
    "tiny_positive": lambda: (_tensor([5e-4, 8e-4, 9e-4, 1e-3, 1.2e-3]),),
    "straddle_eps": lambda: (_tensor([5e-4, 1e-3, 2e-3, -5e-4, -1e-3]),),
    "mixed_range": lambda: (_tensor([1e-4, 5e-4, 2e-3, 1e-2, 5e-2]),),
}


class Log1p(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x)


@common.parametrize("test_data", test_data_suite)
def test_log1p_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](Log1p(), test_data(), aten_op, exir_op)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_log1p_tosa_INT(test_data: input_t1):
    pipeline = TosaPipelineINT[input_t1](
        Log1p(),
        test_data(),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_log1p_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Log1p(),
        test_data(),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_log1p_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Log1p(),
        test_data(),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()
