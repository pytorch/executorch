# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import unittest

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Model(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.hardswish(x, inplace=self.inplace)


@parameterize_by_dtype
def test_hardswish_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(Model(), ((torch.rand(2, 10)).to(dtype),))


def test_hardswish_f32_single_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(20),))


def test_hardswish_f32_multi_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(2, 3, 4, 5),))


@unittest.skip("In place activations aren't properly defunctionalized yet.")
def test_hardswish_f32_inplace(test_runner) -> None:
    test_runner.lower_and_run_model(Model(inplace=True), (torch.randn(3, 4, 5),))


def test_hardswish_f32_boundary_values(test_runner) -> None:
    # Test with values that span the hardswish's piecewise regions
    x = torch.tensor([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    test_runner.lower_and_run_model(Model(), (x,))
