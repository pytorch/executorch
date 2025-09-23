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
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__()
        self.maxpool = torch.nn.MaxPool3d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x):
        return self.maxpool(x)


@parameterize_by_dtype
def test_maxpool3d_dtype(test_runner, dtype) -> None:
    # Input shape: (batch_size, channels, depth, height, width)
    test_runner.lower_and_run_model(
        Model().to(dtype),
        ((torch.rand(1, 4, 8, 8, 8) * 10).to(dtype),),
    )


def test_maxpool3d_kernel_size(test_runner) -> None:
    # Test with different kernel sizes
    test_runner.lower_and_run_model(
        Model(kernel_size=1),
        (torch.randn(1, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(kernel_size=(1, 2, 2)),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_maxpool3d_stride(test_runner) -> None:
    # Test with different stride values
    test_runner.lower_and_run_model(
        Model(stride=2),
        (torch.randn(1, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(stride=(1, 2, 2)),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_maxpool3d_padding(test_runner) -> None:
    # Test with different padding values
    test_runner.lower_and_run_model(
        Model(padding=1),
        (torch.randn(1, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(padding=(0, 1, 1)),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_maxpool3d_dilation(test_runner) -> None:
    # Test with different dilation values
    test_runner.lower_and_run_model(
        Model(dilation=2),
        (torch.randn(1, 4, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(dilation=(1, 2, 2)),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_maxpool3d_ceil_mode(test_runner) -> None:
    # Test with ceil_mode=True
    test_runner.lower_and_run_model(
        Model(ceil_mode=True),
        (torch.randn(1, 4, 8, 8, 8),),
    )


def test_maxpool3d_return_indices(test_runner) -> None:
    # Test with return_indices=True
    class ModelWithIndices(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = torch.nn.MaxPool3d(
                kernel_size=3,
                stride=2,
                padding=1,
                return_indices=True,
            )

        def forward(self, x):
            # Return both output and indices
            return self.maxpool(x)

    # Create a test input tensor
    input_tensor = torch.randn(1, 4, 8, 8, 8)

    test_runner.lower_and_run_model(
        Model(kernel_size=3, stride=2, padding=1),
        (input_tensor,),
    )


def test_maxpool3d_batch_sizes(test_runner) -> None:
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


def test_maxpool3d_input_sizes(test_runner) -> None:
    # Test with different input sizes
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 2, 8, 8, 8),),
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(1, 8, 8, 8, 8),),
    )


def test_maxpool3d_combinations(test_runner) -> None:
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
        Model(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1), dilation=2),
        (torch.randn(1, 4, 8, 10, 10),),
    )
