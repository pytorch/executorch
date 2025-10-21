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

input_t = tuple[torch.Tensor]
test_data_t = tuple[torch.nn.Module, input_t]


class Unflatten(torch.nn.Module):
    aten_op: str = "torch.ops.aten.unflatten.int"

    def __init__(self, dim: int, sizes: Tuple[int, ...]):
        super().__init__()
        self.dim = dim
        self.sizes = sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.unflatten(x, self.dim, self.sizes)

    test_data: dict[str, test_data_t] = {
        "rand_3d_batch3": (lambda: (Unflatten(1, (-1, 2)), (torch.rand(3, 4, 4),))),
        "rand_3d_batch1": (lambda: (Unflatten(1, (-1, 2)), (torch.rand(1, 4, 4),))),
        "randn_4d_dim1": (lambda: (Unflatten(1, (2, 2)), (torch.randn(3, 4, 5, 1),))),
        "randn_4d_dim3": (lambda: (Unflatten(3, (2, 2)), (torch.randn(1, 1, 5, 4),))),
    }


@common.parametrize("test_data", Unflatten.test_data)
def test_unflatten_int_tosa_FP(test_data: test_data_t):
    module, inputs = test_data()
    pipeline = TosaPipelineFP[input_t](
        module,
        inputs,
        Unflatten.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Unflatten.test_data)
def test_unflatten_int_tosa_INT(test_data: test_data_t):
    module, inputs = test_data()
    pipeline = TosaPipelineINT[input_t](module, inputs, Unflatten.aten_op)
    pipeline.run()


@common.parametrize("test_data", Unflatten.test_data, strict=False)
@common.XfailIfNoCorstone300
def test_unflatten_int_u55_INT(test_data: test_data_t):
    module, inputs = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        module,
        inputs,
        Unflatten.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Unflatten.test_data, strict=False)
@common.XfailIfNoCorstone320
def test_unflatten_int_u85_INT(test_data: test_data_t):
    module, inputs = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        module,
        inputs,
        Unflatten.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Unflatten.test_data)
@common.SkipIfNoModelConverter
def test_unflatten_int_vgf_FP(test_data: test_data_t):
    module, inputs = test_data()
    pipeline = VgfPipeline[input_t](
        module,
        inputs,
        Unflatten.aten_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@common.parametrize("test_data", Unflatten.test_data)
@common.SkipIfNoModelConverter
def test_unflatten_int_vgf_INT(test_data: test_data_t):
    module, inputs = test_data()
    pipeline = VgfPipeline[input_t](
        module,
        inputs,
        Unflatten.aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
