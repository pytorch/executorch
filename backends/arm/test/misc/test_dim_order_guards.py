# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)


input_t1 = Tuple[torch.Tensor]  # Input x


class Conv2D(torch.nn.Module):
    inputs: dict[str, input_t1] = {
        "randn": (torch.randn(1, 2, 20, 20).to(memory_format=torch.channels_last),),
    }

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 3))

    def forward(self, x):
        return self.conv2d(x)


@common.parametrize("test_data", Conv2D.inputs)
def test_tosa_FP_pipeline(test_data: input_t1):
    module = Conv2D()
    pipeline = TosaPipelineFP[input_t1](
        module,
        test_data,
        [],
        [],
        use_to_edge_transform_and_lower=False,
    )
    pos = pipeline.find_pos("partition")
    pipeline._stages = pipeline._stages[:pos]
    pipeline.run()
    with pytest.raises(RuntimeError):
        pipeline.tester.partition()


@common.parametrize("test_data", Conv2D.inputs)
def test_tosa_INT_pipeline(test_data: input_t1):
    module = Conv2D()
    pipeline = TosaPipelineINT[input_t1](
        module,
        test_data,
        [],
        [],
        use_to_edge_transform_and_lower=False,
    )
    pos = pipeline.find_pos("partition")
    pipeline._stages = pipeline._stages[:pos]
    pipeline.run()
    with pytest.raises(RuntimeError):
        pipeline.tester.partition()
