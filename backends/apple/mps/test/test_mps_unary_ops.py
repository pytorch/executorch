#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import inspect

import torch
from executorch.backends.apple.mps.test.test_mps_utils import TestMPS


class TestMPSLoigcal(TestMPS):
    def test_mps_logical_not(self):
        class LogicalNot(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.logical_not()

        module = LogicalNot()
        model_inputs = (torch.tensor([1, 1, 0, 0], dtype=torch.bool),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )
