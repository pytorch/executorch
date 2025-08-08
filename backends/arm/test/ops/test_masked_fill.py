# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


aten_op = "torch.aten.ops.masked_fill.Scalar"
exir_op = "executorch_exir_dialects_edge__ops_aten_masked_fill_scalar"

input_t = Tuple[torch.Tensor, torch.Tensor, float]


class MaskedFill(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, value: float
    ) -> torch.Tensor:
        return torch.masked_fill(x, mask, value)


test_modules = {
    "masked_fill_1": lambda: (
        MaskedFill(),
        (
            torch.rand(1, 3, 4, 5),
            (torch.rand(1, 3, 4, 5) < 0.5),  # boolean mask
            -1.0,
        ),
    ),
    "masked_fill_2": lambda: (
        MaskedFill(),
        (
            torch.rand(1, 10, 10, 10),
            (torch.rand(1, 10, 10, 10) > 0.75),
            3.14,
        ),
    ),
    "masked_fill_3_zero_fill": lambda: (
        MaskedFill(),
        (
            torch.rand(1, 3, 4, 5),
            torch.rand(1, 3, 4, 5) < 0.2,
            0.0,
        ),
    ),
    "masked_fill_4_full_mask": lambda: (
        MaskedFill(),
        (
            torch.rand(1, 3, 4, 5),
            torch.ones(1, 3, 4, 5, dtype=torch.bool),
            7.0,
        ),
    ),
    "masked_fill_5_no_mask": lambda: (
        MaskedFill(),
        (
            torch.rand(1, 3, 4, 5),
            torch.zeros(1, 3, 4, 5, dtype=torch.bool),
            -3.0,
        ),
    ),
    "masked_fill_6_scalar_broadcast": lambda: (
        MaskedFill(),
        (
            torch.rand(1, 1, 1, 1),
            torch.tensor([[[[True]]]]),
            42.0,
        ),
    ),
    "masked_fill_7_large_tensor": lambda: (
        MaskedFill(),
        (
            torch.rand(1, 8, 8, 8),
            torch.rand(1, 8, 8, 8) > 0.5,
            -127.0,
        ),
    ),
    "masked_fill_8_extreme_scalar_inf": lambda: (
        MaskedFill(),
        (
            torch.rand(1, 3, 7, 5),
            torch.rand(1, 3, 7, 5) > 0.5,
            float("inf"),
        ),
    ),
}


@common.parametrize("test_module", test_modules)
def test_masked_fill_scalar_tosa_FP(test_module):
    module, inputs = test_module()
    pipeline = TosaPipelineFP[input_t](module, inputs, aten_op=[])
    pipeline.run()


@common.parametrize("test_module", test_modules)
def test_masked_fill_scalar_tosa_INT(test_module):
    module, inputs = test_module()
    pipeline = TosaPipelineINT[input_t](
        module,
        inputs,
        aten_op=[],
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone300
def test_masked_fill_scalar_u55_INT(test_module):
    module, inputs = test_module()
    pipeline = OpNotSupportedPipeline[input_t](
        module,
        inputs,
        {exir_op: 0, "executorch_exir_dialects_edge__ops_aten_where_self": 1},
        n_expected_delegates=0,
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.XfailIfNoCorstone320
def test_masked_fill_scalar_u85_INT(test_module):
    module, inputs = test_module()
    pipeline = EthosU85PipelineINT[input_t](
        module,
        inputs,
        aten_ops=[],
        exir_ops=exir_op,
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.SkipIfNoModelConverter
def test_masked_fill_scalar_vgf_FP(test_module):
    module, inputs = test_module()
    pipeline = VgfPipeline[input_t](
        module, inputs, aten_op=[], tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize("test_module", test_modules)
@common.SkipIfNoModelConverter
def test_masked_fill_scalar_vgf_INT(test_module):
    module, inputs = test_module()
    pipeline = VgfPipeline[input_t](
        module, inputs, aten_op=[], tosa_version="TOSA-1.0+INT"
    )
    pipeline.run()
