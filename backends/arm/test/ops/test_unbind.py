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

input_t = tuple[torch.Tensor]
test_data_t = tuple[int, torch.dtype]


class Unbind(torch.nn.Module):
    aten_op: str = "torch.ops.aten.unbind.int"

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return torch.unbind(x, self.dim)

    test_data: dict[str, test_data_t] = {
        "randn_4d": (lambda: (torch.randn(1, 5, 4, 3),), (2,)),
        "randn_3d": (lambda: (torch.randn(5, 4, 3),), (0,)),
    }


@common.parametrize("test_data", Unbind.test_data)
def test_unbind_int_tosa_FP(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t](
        Unbind(*init_data),
        input_data(),
        Unbind.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Unbind.test_data)
def test_unbind_int_tosa_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineINT[input_t](
        Unbind(*init_data),
        input_data(),
        Unbind.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Unbind.test_data)
@common.SkipIfNoModelConverter
def test_unbind_int_vgf_FP(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = VgfPipeline[input_t](
        Unbind(*init_data),
        input_data(),
        Unbind.aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Unbind.test_data)
@common.SkipIfNoModelConverter
def test_unbind_int_vgf_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = VgfPipeline[input_t](
        Unbind(*init_data),
        input_data(),
        Unbind.aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
