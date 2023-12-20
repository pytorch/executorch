# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.test_xnnpack_utils import TestXNNPACK

from torch.ao.quantization.observer import (
    per_channel_weight_observer_range_neg_127_to_127,
    weight_observer_range_neg_127_to_127,
)


class TestXNNPACKQuantized(TestXNNPACK):
    # TODO(T158652796)
    @unittest.expectedFailure
    def test_xnnpack_qelu(self):
        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.elu = torch.nn.ELU(alpha=0.5)

            def forward(self, x):
                return self.elu(x)

        example_inputs = (torch.randn(1, 3, 4, 4),)
        self.quantize_and_test_model(ELUModule(), example_inputs)

    # TODO(T158652796)
    @unittest.expectedFailure
    def test_xnnpack_qelu2(self):
        class ELUModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.elu(x, alpha=1.2)

        example_inputs = (torch.randn(1, 3, 4, 4),)
        self.quantize_and_test_model(ELUModule(), example_inputs)
