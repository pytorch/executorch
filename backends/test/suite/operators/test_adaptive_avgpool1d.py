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
        output_size=5,
    ):
        super().__init__()
        self.adaptive_avgpool = torch.nn.AdaptiveAvgPool1d(
            output_size=output_size,
        )

    def forward(self, x):
        return self.adaptive_avgpool(x)


@parameterize_by_dtype
def test_adaptive_avgpool1d_dtype(test_runner, dtype) -> None:
    # Input shape: (batch_size, channels, length)
    test_runner.lower_and_run_model(
        Model().to(dtype),
        ((torch.rand(1, 8, 100) * 10).to(dtype),),
    )


def test_adaptive_avgpool1d_output_size(test_runner) -> None:
    # Test with different output sizes
    test_runner.lower_and_run_model(
        Model(output_size=1),
        (torch.randn(1, 8, 100),),
    )
    test_runner.lower_and_run_model(
        Model(output_size=10),
        (torch.randn(1, 8, 100),),
    )
    test_runner.lower_and_run_model(
        Model(output_size=50),
        (torch.randn(1, 8, 100),),
    )


def test_adaptive_avgpool1d_batch_sizes(test_runner) -> None:
    # Test with batch inputs
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(2, 8, 100),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(8, 8, 100),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(16, 8, 100),),
    )


def test_adaptive_avgpool1d_input_sizes(test_runner) -> None:
    # Test with different input sizes
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 4, 100),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 16, 100),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 8, 50),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 8, 200),),
    )
