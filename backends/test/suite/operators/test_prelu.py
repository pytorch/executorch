# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Model(torch.nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.prelu = torch.nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        return self.prelu(x)


@parameterize_by_dtype
def test_prelu_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model().to(dtype), ((torch.rand(2, 10) * 2 - 1).to(dtype),)
    )


def test_prelu_f32_single_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(20),))


def test_prelu_f32_multi_dim(test_runner) -> None:
    test_runner.lower_and_run_model(Model(), (torch.randn(2, 3, 4, 5),))


def test_prelu_f32_custom_init(test_runner) -> None:
    test_runner.lower_and_run_model(Model(init=0.1), (torch.randn(3, 4, 5),))


def test_prelu_f32_channel_shared(test_runner) -> None:
    # Default num_parameters=1 means the parameter is shared across all channels
    test_runner.lower_and_run_model(Model(num_parameters=1), (torch.randn(2, 3, 4, 5),))


def test_prelu_f32_per_channel_parameter(test_runner) -> None:
    # num_parameters=3 means each channel has its own parameter (for dim=1)
    test_runner.lower_and_run_model(Model(num_parameters=3), (torch.randn(2, 3, 4, 5),))


def test_prelu_f32_boundary_values(test_runner) -> None:
    # Test with specific positive and negative values
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    test_runner.lower_and_run_model(Model(), (x,))
