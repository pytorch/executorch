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
    def __init__(self, threshold=0.0, value=0.0, inplace=False):
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.threshold(
            x, threshold=self.threshold, value=self.value, inplace=self.inplace
        )


@parameterize_by_dtype
def test_threshold_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(Model(), ((torch.rand(2, 10) * 10 - 5).to(dtype),))


def test_threshold_f32_single_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(20),))


def test_threshold_f32_multi_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(2, 3, 4, 5),))


def test_threshold_f32_custom_threshold(test_runner) -> None:
    test_runner.lower_and_run_model(Model(threshold=1.0), (torch.randn(3, 4, 5),))


def test_threshold_f32_custom_value(test_runner) -> None:
    test_runner.lower_and_run_model(Model(value=2.0), (torch.randn(3, 4, 5),))


def test_threshold_f32_custom_threshold_value(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(threshold=0.5, value=1.0), (torch.randn(3, 4, 5),)
    )


@unittest.skip("In place activations aren't properly defunctionalized yet.")
def test_threshold_f32_inplace(test_runner) -> None:
    test_runner.lower_and_run_model(Model(inplace=True), (torch.randn(3, 4, 5),))


def test_threshold_f32_boundary_values(test_runner) -> None:
    # Test with specific values around the threshold
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    test_runner.lower_and_run_model(Model(), (x,))


def test_threshold_f32_all_params(test_runner) -> None:
    # Test with all parameters customized
    test_runner.lower_and_run_model(
        Model(threshold=0.5, value=3.0, inplace=True),
        (torch.randn(3, 4, 5),),
    )
