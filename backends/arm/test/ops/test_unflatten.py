# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineBI,
    TosaPipelineMI,
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
        "randn_4d": (lambda: (Unflatten(1, (2, 2)), (torch.randn(3, 4, 5, 1),))),
        "rand_3d": (lambda: (Unflatten(1, (-1, 2)), (torch.rand(3, 4, 4),))),
    }


@common.parametrize("test_data", Unflatten.test_data)
def test_unflatten_int_tosa_MI(test_data: test_data_t):
    module, inputs = test_data()
    pipeline = TosaPipelineMI[input_t](
        module,
        inputs,
        Unflatten.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", Unflatten.test_data)
def test_unflatten_int_tosa_BI(test_data: test_data_t):
    module, inputs = test_data()
    pipeline = TosaPipelineBI[input_t](
        module,
        inputs,
        Unflatten.aten_op,
    )
    pipeline.run()
