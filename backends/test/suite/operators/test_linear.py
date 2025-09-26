# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Model(torch.nn.Module):
    def __init__(
        self,
        in_features=67,
        out_features=43,
        bias=True,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        return self.linear(x)


@parameterize_by_dtype
def test_linear_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model().to(dtype),
        ((torch.rand(16, 64) * 10).to(dtype),),
    )


@parameterize_by_dtype
def test_linear_no_bias_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model(bias=False).to(dtype),
        ((torch.rand(16, 64) * 10).to(dtype),),
    )


def test_linear_feature_sizes(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(in_features=32, out_features=16),
        (torch.randn(20, 32),),
    )
    test_runner.lower_and_run_model(
        Model(in_features=128, out_features=64),
        (torch.randn(8, 128),),
    )
    test_runner.lower_and_run_model(
        Model(in_features=256, out_features=1),
        (torch.randn(4, 256),),
    )
    test_runner.lower_and_run_model(
        Model(in_features=1, out_features=512),
        (torch.randn(1024, 1),),
    )


def test_linear_no_bias(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(bias=False),
        (torch.randn(16, 64),),
    )
    test_runner.lower_and_run_model(
        Model(in_features=128, out_features=96, bias=False),
        (torch.randn(8, 128),),
    )


def test_linear_batch_sizes(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(8, 64),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(32, 64),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(100, 64),),
    )


def test_linear_unbatched(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(in_features=512),
        (torch.randn(512),),
    )


def test_linear_leading_batch(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(4, 8, 64),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(2, 4, 8, 64),),
    )
