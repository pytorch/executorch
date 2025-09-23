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
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.elu(x, alpha=self.alpha, inplace=self.inplace)


@parameterize_by_dtype
def test_elu_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(Model(), ((torch.rand(2, 10) * 100).to(dtype),))


def test_elu_f32_single_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(20),))


def test_elu_f32_multi_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(2, 3, 4, 5),))


def test_elu_f32_alpha(test_runner) -> None:
    test_runner.lower_and_run_model(Model(alpha=0.5), (torch.randn(3, 4, 5),))


@unittest.skip("In place activations aren't properly defunctionalized yet.")
def test_elu_f32_inplace(test_runner) -> None:
    test_runner.lower_and_run_model(Model(inplace=True), (torch.randn(3, 4, 5),))
