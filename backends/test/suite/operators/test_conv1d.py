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
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.conv(x)


@parameterize_by_dtype
def test_conv1d_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model().to(dtype),
        ((torch.rand(4, 3, 50) * 10).to(dtype),),
    )


def test_conv1d_basic(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(4, 3, 50),),
    )


def test_conv1d_kernel_size(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(kernel_size=1),
        (torch.randn(4, 3, 50),),
    )
    test_runner.lower_and_run_model(
        Model(kernel_size=5),
        (torch.randn(4, 3, 50),),
    )


def test_conv1d_stride(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(stride=2),
        (torch.randn(4, 3, 50),),
    )


def test_conv1d_padding(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(padding=1),
        (torch.randn(4, 3, 50),),
    )
    test_runner.lower_and_run_model(
        Model(padding=2),
        (torch.randn(4, 3, 50),),
    )


def test_conv1d_dilation(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(dilation=2),
        (torch.randn(4, 3, 50),),
    )


def test_conv1d_groups(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(in_channels=6, out_channels=6, groups=3),
        (torch.randn(4, 6, 50),),
    )


def test_conv1d_depthwise(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(in_channels=8, out_channels=8, groups=8),
        (torch.randn(4, 8, 50),),
    )


def test_conv1d_no_bias(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(bias=False),
        (torch.randn(4, 3, 50),),
    )


def test_conv1d_padding_modes(test_runner) -> None:
    for mode in ["zeros", "reflect", "replicate", "circular"]:
        test_runner.lower_and_run_model(
            Model(padding=1, padding_mode=mode),
            (torch.randn(4, 3, 50),),
        )


def test_conv1d_channels(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(in_channels=1, out_channels=1),
        (torch.randn(4, 1, 50),),
    )
    test_runner.lower_and_run_model(
        Model(in_channels=5, out_channels=10),
        (torch.randn(4, 5, 50),),
    )
