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
        kernel_size=3,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        super().__init__()

        # Create the avgpool layer with the given parameters
        # torch.nn.AvgPool3d accepts both int and tuple types for kernel_size, stride, and padding
        self.avgpool = torch.nn.AvgPool3d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    def forward(self, x):
        return self.avgpool(x)


@parameterize_by_dtype
def test_avgpool3d_dtype(test_runner, dtype) -> None:
    # Input shape: (batch_size, channels, depth, height, width)
    test_runner.lower_and_run_model(
        Model().to(dtype),
        ((torch.rand(1, 4, 8, 8, 8) * 10).to(dtype),),
    )


def test_avgpool3d_kernel_size(test_runner) -> None:
    # Test with different kernel sizes
    test_runner.lower_and_run_model(
        Model(kernel_size=1),
        (torch.randn(1, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(kernel_size=(1, 2, 2)),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_avgpool3d_stride(test_runner) -> None:
    # Test with different stride values
    test_runner.lower_and_run_model(
        Model(stride=2),
        (torch.randn(1, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(stride=(1, 2, 2)),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_avgpool3d_padding(test_runner) -> None:
    # Test with different padding values
    test_runner.lower_and_run_model(
        Model(padding=1),
        (torch.randn(1, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(padding=(0, 1, 1)),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_avgpool3d_ceil_mode(test_runner) -> None:
    # Test with ceil_mode=True
    test_runner.lower_and_run_model(
        Model(ceil_mode=True),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_avgpool3d_count_include_pad(test_runner) -> None:
    # Test with count_include_pad=False
    test_runner.lower_and_run_model(
        Model(padding=1, count_include_pad=False),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_avgpool3d_batch_sizes(test_runner) -> None:
    # Test with batch inputs
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(2, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(8, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(16, 4, 8, 8, 8),),
    )


def test_avgpool3d_input_sizes(test_runner) -> None:
    # Test with different input sizes
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 2, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 8, 8, 8, 8),),
    )


def test_avgpool3d_combinations(test_runner) -> None:
    # Test with combinations of parameters
    test_runner.lower_and_run_model(
        Model(kernel_size=2, stride=2, padding=1),
        (torch.randn(1, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        (torch.randn(1, 4, 10, 10, 10),),
    )
    test_runner.lower_and_run_model(
        Model(
            kernel_size=(2, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            count_include_pad=False,
        ),
        (torch.randn(1, 4, 8, 10, 10),),
    )
