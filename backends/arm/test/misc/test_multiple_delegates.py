# Copyright 2024-2025 Arm Limited and/or its affiliates.
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


input_t1 = Tuple[torch.Tensor, torch.Tensor]  # Input x, y


class MultipleDelegatesModule(torch.nn.Module):
    inputs = {
        "randn": (torch.randn(10, 4, 5), torch.randn(10, 4, 5)),
    }

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = x + y
        s = torch.tan(z)
        return s * z


@common.parametrize("test_data", MultipleDelegatesModule.inputs)
def test_tosa_MI_pipeline(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](MultipleDelegatesModule(), test_data, [], [])
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 2}
    )
    pipeline.run()


@common.parametrize("test_data", MultipleDelegatesModule.inputs)
def test_tosa_BI_pipeline(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](
        MultipleDelegatesModule(), test_data, [], [], qtol=1
    )
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 2}
    )
    pipeline.run()
