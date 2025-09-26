# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Tuple, Union

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


class Model(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=6,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv_transpose(x)


@parameterize_by_dtype
def test_convtranspose2d_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model().to(dtype),
        ((torch.rand(4, 3, 16, 16) * 10).to(dtype),),
    )


def test_convtranspose2d_basic(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(4, 3, 16, 16),),
    )


def test_convtranspose2d_kernel_size(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(kernel_size=1),
        (torch.randn(4, 3, 16, 16),),
    )
    test_runner.lower_and_run_model(
        Model(kernel_size=5),
        (torch.randn(4, 3, 16, 16),),
    )
    test_runner.lower_and_run_model(
        Model(kernel_size=(3, 5)),
        (torch.randn(4, 3, 16, 16),),
    )


def test_convtranspose2d_stride(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(stride=2),
        (torch.randn(4, 3, 16, 16),),
    )
    test_runner.lower_and_run_model(
        Model(stride=(2, 1)),
        (torch.randn(4, 3, 16, 16),),
    )


def test_convtranspose2d_padding(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(padding=1),
        (torch.randn(4, 3, 16, 16),),
    )
    test_runner.lower_and_run_model(
        Model(padding=(1, 2)),
        (torch.randn(4, 3, 16, 16),),
    )


def test_convtranspose2d_output_padding(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(stride=2, output_padding=1),
        (torch.randn(4, 3, 16, 16),),
    )
    test_runner.lower_and_run_model(
        Model(stride=(2, 2), output_padding=(1, 0)),
        (torch.randn(4, 3, 16, 16),),
    )


def test_convtranspose2d_dilation(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(dilation=2),
        (torch.randn(4, 3, 16, 16),),
    )
    test_runner.lower_and_run_model(
        Model(dilation=(2, 1)),
        (torch.randn(4, 3, 16, 16),),
    )


def test_convtranspose2d_groups(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(in_channels=6, out_channels=6, groups=3),
        (torch.randn(4, 6, 16, 16),),
    )


def test_convtranspose2d_depthwise(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(in_channels=8, out_channels=8, groups=8),
        (torch.randn(4, 8, 16, 16),),
    )


def test_convtranspose2d_no_bias(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(bias=False),
        (torch.randn(4, 3, 16, 16),),
    )


def test_convtranspose2d_channels(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(in_channels=1, out_channels=1),
        (torch.randn(4, 1, 16, 16),),
    )
    test_runner.lower_and_run_model(
        Model(in_channels=5, out_channels=10),
        (torch.randn(4, 5, 16, 16),),
    )


def test_convtranspose2d_different_spatial_dims(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (torch.randn(4, 3, 20, 16),),
    )
