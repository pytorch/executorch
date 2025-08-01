# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from executorch.backends.arm.test.common import parametrize
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
)


class NonPersistentBuffer(nn.Module):
    """
    Min code version registering a non-persistent input buffer.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("test_buff", torch.rand(2, 2, 2, 2), persistent=False)

    def forward(self, x):
        return x - self.test_buff


test_input = {"input": (torch.ones(2, 2, 2, 2),)}

input_t = tuple[torch.Tensor]


@parametrize("test_data", test_input)
def test_non_persistent_buffer_FP(test_data: input_t):
    """
    Test validates Arm backend handling of non-persistent buffers
    and ensures that there are no asserts or errors when they are used.
    """
    TosaPipelineFP[input_t](NonPersistentBuffer(), test_data, "").run()


@parametrize("test_data", test_input)
def test_non_persistent_buffer_INT(test_data: input_t):
    """
    Test validates Arm backend handling of non-persistent buffers
    and ensures that there are no asserts or errors when they are used.
    """
    TosaPipelineINT[input_t](NonPersistentBuffer(), test_data, "").run()
