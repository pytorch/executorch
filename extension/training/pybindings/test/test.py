# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.exir import to_edge

from executorch.extension.training import (
    _load_for_executorch_for_training_from_buffer,
    get_sgd_optimizer,
)
from torch.export.experimental import _export_forward_backward


class TestTraining(unittest.TestCase):
    class ModuleSimpleTrain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, x, y):
            return self.loss(self.linear(x).softmax(dim=0), y)

        def get_random_inputs(self):
            return (torch.randn(3), torch.tensor([1.0, 0.0, 0.0]))

    def test(self):
        m = self.ModuleSimpleTrain()
        ep = torch.export.export(m, m.get_random_inputs())
        ep = _export_forward_backward(ep)
        ep = to_edge(ep)
        ep = ep.to_executorch()
        buffer = ep.buffer
        tm = _load_for_executorch_for_training_from_buffer(buffer)

        tm.forward_backward("forward", m.get_random_inputs())
        orig_param = list(tm.named_parameters().values())[0].clone()
        optimizer = get_sgd_optimizer(
            tm.named_parameters(),
            0.1,
            0,
            0,
            0,
            False,
        )
        optimizer.step(tm.named_gradients())
        self.assertFalse(
            torch.allclose(orig_param, list(tm.named_parameters().values())[0])
        )
